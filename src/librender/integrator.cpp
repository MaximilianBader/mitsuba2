#include <thread>
#include <mutex>

#include <enoki/morton.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

NAMESPACE_BEGIN(mitsuba)

// -----------------------------------------------------------------------------

MTS_VARIANT SamplingIntegrator<Float, Spectrum>::SamplingIntegrator(const Properties &props)
    : Base(props) {

    m_block_size = (uint32_t) props.size_("block_size", 0);
    uint32_t block_size = math::round_to_power_of_two(m_block_size);
    if (m_block_size > 0 && block_size != m_block_size) {
        Log(Warn, "Setting block size from %i to next higher power of two: %i", m_block_size,
            block_size);
        m_block_size = block_size;
    }

    m_samples_per_pass = (uint32_t) props.size_("samples_per_pass", (size_t) -1);
    m_timeout = props.float_("timeout", -1.f);

    /// Disable direct visibility of emitters if needed
    m_hide_emitters = props.bool_("hide_emitters", false);
}

MTS_VARIANT SamplingIntegrator<Float, Spectrum>::~SamplingIntegrator() { }

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::cancel() {
    m_stop = true;
}

MTS_VARIANT std::vector<std::string> SamplingIntegrator<Float, Spectrum>::aov_names() const {
    return { };
}

MTS_VARIANT bool SamplingIntegrator<Float, Spectrum>::render(Scene *scene, Sensor *sensor) {
    ScopedPhase sp(ProfilerPhase::Render);
    m_stop = false;

    ref<Film> film = sensor->film();
    ScalarVector2i film_size = film->crop_size();

    size_t total_spp        = sensor->sampler()->sample_count();
    size_t samples_per_pass = (m_samples_per_pass == (size_t) -1)
                               ? total_spp : std::min((size_t) m_samples_per_pass, total_spp);
    if ((total_spp % samples_per_pass) != 0)
        Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
              total_spp, samples_per_pass);

    size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

    std::vector<std::string> channels = aov_names();
    bool has_aovs = !channels.empty();

    // Insert default channels and set up the film
    for (size_t i = 0; i < 5; ++i)
        channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
    film->prepare(channels);

    m_render_timer.reset();
    if constexpr (!is_cuda_array_v<Float>) {
        /// Render on the CPU using a spiral pattern
        size_t n_threads = __global_thread_count;
        Log(Info, "Starting render job (%ix%i, %i sample%s,%s %i thread%s)",
            film_size.x(), film_size.y(),
            total_spp, total_spp == 1 ? "" : "s",
            n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
            n_threads, n_threads == 1 ? "" : "s");

        if (m_timeout > 0.f)
            Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

        // Find a good block size to use for splitting up the total workload.
        if (m_block_size == 0) {
            uint32_t block_size = MTS_BLOCK_SIZE;
            while (true) {
                if (block_size == 1 || hprod((film_size + block_size - 1) / block_size) >= n_threads)
                    break;
                block_size /= 2;
            }
            m_block_size = block_size;
        }

        Spiral spiral(film, m_block_size, n_passes);

        ThreadEnvironment env;
        ref<ProgressReporter> progress = new ProgressReporter("Rendering");
        std::mutex mutex;

        // Total number of blocks to be handled, including multiple passes.
        size_t total_blocks = spiral.block_count() * n_passes,
               blocks_done = 0;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, total_blocks, 1),
            [&](const tbb::blocked_range<size_t> &range) {
                ScopedSetThreadEnvironment set_env(env);
                ref<Sampler> sampler = sensor->sampler()->clone();
                ref<ImageBlock> block = new ImageBlock(m_block_size, channels.size(),
                                                       film->reconstruction_filter(),
                                                       !has_aovs);
                scoped_flush_denormals flush_denormals(true);
                std::unique_ptr<Float[]> aovs(new Float[channels.size()]);

                // For each block
                for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                    auto [offset, size, block_id] = spiral.next_block();
                    Assert(hprod(size) != 0);
                    block->set_size(size);
                    block->set_offset(offset);

                    render_block(scene, sensor, sampler, block,
                                 aovs.get(), samples_per_pass, block_id);

                    film->put(block);

                    /* Critical section: update progress bar */ {
                        std::lock_guard<std::mutex> lock(mutex);
                        blocks_done++;
                        progress->update(blocks_done / (ScalarFloat) total_blocks);
                    }
                }
            }
        );
    } else {
        Log(Info, "Start rendering...");

        ref<Sampler> sampler = sensor->sampler();
        sampler->set_samples_per_wavefront((uint32_t) samples_per_pass);

        ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());
        ScalarUInt32 wavefront_size = hprod(film_size) * (uint32_t) samples_per_pass;
        if (sampler->wavefront_size() != wavefront_size)
            sampler->seed(0, wavefront_size);

        UInt32 idx = arange<UInt32>(wavefront_size);
        if (samples_per_pass != 1)
            idx /= (uint32_t) samples_per_pass;

        ref<ImageBlock> block = new ImageBlock(film_size, channels.size(),
                                               film->reconstruction_filter(),
                                               !has_aovs);
        block->clear();
        block->set_offset(sensor->film()->crop_offset());

        Vector2f pos = Vector2f(Float(idx % uint32_t(film_size[0])),
                                Float(idx / uint32_t(film_size[0])));
        pos += block->offset();

        std::vector<Float> aovs(channels.size());

        for (size_t i = 0; i < n_passes; i++)
            render_sample(scene, sensor, sampler, block, aovs.data(),
                          pos, diff_scale_factor);

        film->put(block);
    }

    if (!m_stop)
        Log(Info, "Rendering finished. (took %s)",
            util::time_string(m_render_timer.value(), true));

    return !m_stop;
}

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::render_block(const Scene *scene,
                                                                   const Sensor *sensor,
                                                                   Sampler *sampler,
                                                                   ImageBlock *block,
                                                                   Float *aovs,
                                                                   size_t sample_count_,
                                                                   size_t block_id) const {
    block->clear();
    uint32_t pixel_count  = (uint32_t)(m_block_size * m_block_size),
             sample_count = (uint32_t)(sample_count_ == (size_t) -1
                                           ? sampler->sample_count()
                                           : sample_count_);

    ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());

    if constexpr (!is_array_v<Float>) {
        for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
            sampler->seed(block_id * pixel_count + i);

            ScalarPoint2u pos = enoki::morton_decode<ScalarPoint2u>(i);
            if (any(pos >= block->size()))
                continue;

            pos += block->offset();
            for (uint32_t j = 0; j < sample_count && !should_stop(); ++j) {
                render_sample(scene, sensor, sampler, block, aovs,
                              pos, diff_scale_factor);
            }
        }
    } else if constexpr (is_array_v<Float> && !is_cuda_array_v<Float>) {
        // Ensure that the sample generation is fully deterministic
        sampler->seed(block_id);

        for (auto [index, active] : range<UInt32>(pixel_count * sample_count)) {
            if (should_stop())
                break;
            Point2u pos = enoki::morton_decode<Point2u>(index / UInt32(sample_count));
            active &= !any(pos >= block->size());
            pos += block->offset();
            render_sample(scene, sensor, sampler, block, aovs, pos, diff_scale_factor, active);
        }
    } else {
        ENOKI_MARK_USED(scene);
        ENOKI_MARK_USED(sensor);
        ENOKI_MARK_USED(aovs);
        ENOKI_MARK_USED(diff_scale_factor);
        ENOKI_MARK_USED(pixel_count);
        ENOKI_MARK_USED(sample_count);
        Throw("Not implemented for CUDA arrays.");
    }
}

MTS_VARIANT void
SamplingIntegrator<Float, Spectrum>::render_sample(const Scene *scene,
                                                   const Sensor *sensor,
                                                   Sampler *sampler,
                                                   ImageBlock *block,
                                                   Float *aovs,
                                                   const Vector2f &pos,
                                                   ScalarFloat diff_scale_factor,
                                                   Mask active) const {
    Vector2f position_sample = pos + sampler->next_2d(active);

    Point2f aperture_sample(.5f);
    if (sensor->needs_aperture_sample())
        aperture_sample = sampler->next_2d(active);

    Float time = sensor->shutter_open();
    if (sensor->shutter_open_time() > 0.f)
        time += sampler->next_1d(active) * sensor->shutter_open_time();

    Float wavelength_sample = sampler->next_1d(active);

    Vector2f adjusted_position =
        (position_sample - sensor->film()->crop_offset()) /
        sensor->film()->crop_size();

    auto [ray, ray_weight] = sensor->sample_ray_differential(
        time, wavelength_sample, adjusted_position, aperture_sample);

    ray.scale_differential(diff_scale_factor);

    const Medium *medium = sensor->medium();
    std::pair<Spectrum, Mask> result = sample(scene, sampler, ray, medium, aovs + 5, active);
    result.first = ray_weight * result.first;

    UnpolarizedSpectrum spec_u = depolarize(result.first);

    Color3f xyz;
    if constexpr (is_monochromatic_v<Spectrum>) {
        xyz = spec_u.x();
    } else if constexpr (is_rgb_v<Spectrum>) {
        xyz = srgb_to_xyz(spec_u, active);
    } else {
        static_assert(is_spectral_v<Spectrum>);
        xyz = spectrum_to_xyz(spec_u, ray.wavelengths, active);
    }

    aovs[0] = xyz.x();
    aovs[1] = xyz.y();
    aovs[2] = xyz.z();
    aovs[3] = select(result.second, Float(1.f), Float(0.f));
    aovs[4] = 1.f;

    block->put(position_sample, aovs, active);

    sampler->advance();
}

MTS_VARIANT std::pair<Spectrum, typename SamplingIntegrator<Float, Spectrum>::Mask>
SamplingIntegrator<Float, Spectrum>::sample(const Scene * /* scene */,
                                            Sampler * /* sampler */,
                                            const RayDifferential3f & /* ray */,
                                            const Medium * /* medium */,
                                            Float * /* aovs */,
                                            Mask /* active */) const {
    NotImplementedError("sample");
}

// -----------------------------------------------------------------------------

MTS_VARIANT MonteCarloIntegrator<Float, Spectrum>::MonteCarloIntegrator(const Properties &props)
    : Base(props) {
    /// Depth to begin using russian roulette
    m_rr_depth = props.int_("rr_depth", 5);
    if (m_rr_depth <= 0)
        Throw("\"rr_depth\" must be set to a value greater than zero!");

    /*  Longest visualized path depth (``-1 = infinite``). A value of \c 1 will
        visualize only directly visible light sources. \c 2 will lead to
        single-bounce (direct-only) illumination, and so on. */
    m_max_depth = props.int_("max_depth", -1);
    if (m_max_depth < 0 && m_max_depth != -1)
        Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");
}

MTS_VARIANT MonteCarloIntegrator<Float, Spectrum>::~MonteCarloIntegrator() { }

// -----------------------------------------------------------------------------

MTS_VARIANT PathLengthOriginIntegrator<Float, Spectrum>::PathLengthOriginIntegrator(const Properties &props) : Base(props) { }

MTS_VARIANT PathLengthOriginIntegrator<Float, Spectrum>::~PathLengthOriginIntegrator() { }

MTS_VARIANT bool PathLengthOriginIntegrator<Float, Spectrum>::render_with_length(Scene *scene, Sensor *sensor) {
    ScopedPhase sp(ProfilerPhase::Render);
    m_stop = false;

    ref<Film> film = sensor->film();
    ScalarVector2i film_size = film->crop_size();

    size_t total_spp        = sensor->sampler()->sample_count();
    size_t samples_per_pass = (m_samples_per_pass == (size_t) -1)
                               ? total_spp : std::min((size_t) m_samples_per_pass, total_spp);
    if ((total_spp % samples_per_pass) != 0)
        Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
              total_spp, samples_per_pass);

    size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

    std::vector<std::string> channels = aov_names();
    bool has_aovs = !channels.empty();

    // Insert default channels and set up the film
    for (size_t i = 0; i < 5; ++i)
        channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
    film->prepare(channels);

    m_render_timer.reset();
    if constexpr (!is_cuda_array_v<Float>) {
        /// Render on the CPU using a spiral pattern
        size_t n_threads = __global_thread_count;
        Log(Info, "Starting render job (%ix%i, %i sample%s,%s %i thread%s)",
            film_size.x(), film_size.y(),
            total_spp, total_spp == 1 ? "" : "s",
            n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
            n_threads, n_threads == 1 ? "" : "s");

        if (m_timeout > 0.f)
            Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

        // Find a good block size to use for splitting up the total workload.
        if (m_block_size == 0) {
            uint32_t block_size = MTS_BLOCK_SIZE;
            while (true) {
                if (block_size == 1 || hprod((film_size + block_size - 1) / block_size) >= n_threads)
                    break;
                block_size /= 2;
            }
            m_block_size = block_size;
        }

        Spiral spiral(film, m_block_size, n_passes);

        ThreadEnvironment env;
        ref<ProgressReporter> progress = new ProgressReporter("Rendering");
        std::mutex mutex;

        // Total number of blocks to be handled, including multiple passes.
        size_t total_blocks = spiral.block_count() * n_passes,
               blocks_done = 0;

        // DEBUGGING:
        std::cout << "\nm_samples_per_pass: " << m_samples_per_pass;
        std::cout << "\ntotal_spp: " << total_spp;
        std::cout << "\nsamples_per_pass: " << samples_per_pass;
        std::cout << "\npasses: " << n_passes;
        std::cout << "\total_blocks: " << total_blocks << "\n";

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, total_blocks, 1),
            [&](const tbb::blocked_range<size_t> &range) {
                ScopedSetThreadEnvironment set_env(env);
                ref<Sampler> sampler = sensor->sampler()->clone();
                ref<ImageBlock> block = new ImageBlock(m_block_size, channels.size(),
                                                       film->reconstruction_filter(),
                                                       !has_aovs);
                scoped_flush_denormals flush_denormals(true);
                
                // For each block
                for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                    auto [offset, size, block_id] = spiral.next_block();
                    Assert(hprod(size) != 0);
                    block->set_size(size);
                    block->set_offset(offset);

                    render_block(scene, sensor, sampler, block,
                                 samples_per_pass, block_id);

                    film->put(block);

                    /* Critical section: update progress bar */ {
                        std::lock_guard<std::mutex> lock(mutex);
                        blocks_done++;
                        progress->update(blocks_done / (ScalarFloat) total_blocks);
                    }
                }
            }
        );
    } else {
        Log(Info, "Start rendering...");

        ref<Sampler> sampler = sensor->sampler();
        sampler->set_samples_per_wavefront((uint32_t) samples_per_pass);

        ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());
        ScalarUInt32 wavefront_size = hprod(film_size) * (uint32_t) samples_per_pass;
        if (sampler->wavefront_size() != wavefront_size)
            sampler->seed(0, wavefront_size);

        UInt32 idx = arange<UInt32>(wavefront_size);
        if (samples_per_pass != 1)
            idx /= (uint32_t) samples_per_pass;

        ref<ImageBlock> block = new ImageBlock(film_size, channels.size(),
                                               film->reconstruction_filter(),
                                               !has_aovs);
        block->clear();
        block->set_offset(sensor->film()->crop_offset());

        Vector2f pos = Vector2f(Float(idx % uint32_t(film_size[0])),
                                Float(idx / uint32_t(film_size[0])));
        pos += block->offset();

        for (size_t i = 0; i < n_passes; i++)
            render_sample(scene, sensor, sampler, block,
                          pos, diff_scale_factor);

        film->put(block);
    }

    if (!m_stop)
        Log(Info, "Rendering finished. (took %s)",
            util::time_string(m_render_timer.value(), true));

    return !m_stop;
}

MTS_VARIANT void PathLengthOriginIntegrator<Float, Spectrum>::render_block(const Scene *scene,
                                                                           const Sensor *sensor,
                                                                           Sampler *sampler,
                                                                   ImageBlock *block,
                                                                   size_t sample_count_,
                                                                   size_t block_id) const {
    block->clear();
    uint32_t pixel_count  = (uint32_t)(m_block_size * m_block_size),
             sample_count = (uint32_t)(sample_count_ == (size_t) -1
                                           ? sampler->sample_count()
                                           : sample_count_);

    ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());

    if constexpr (!is_array_v<Float>) {
        for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
            sampler->seed(block_id * pixel_count + i);

            ScalarPoint2u pos = enoki::morton_decode<ScalarPoint2u>(i);
            if (any(pos >= block->size()))
                continue;

            pos += block->offset();
            for (uint32_t j = 0; j < sample_count && !should_stop(); ++j) {
                render_sample(scene, sensor, sampler, block,
                              pos, diff_scale_factor);
            }
        }
    } else if constexpr (is_array_v<Float> && !is_cuda_array_v<Float>) {
        // Ensure that the sample generation is fully deterministic
        sampler->seed(block_id);

        for (auto [index, active] : range<UInt32>(pixel_count * sample_count)) {
            if (should_stop())
                break;
            Point2u pos = enoki::morton_decode<Point2u>(index / UInt32(sample_count));
            active &= !any(pos >= block->size());
            pos += block->offset();
            render_sample(scene, sensor, sampler, block, pos, diff_scale_factor, active);
        }
    } else {
        ENOKI_MARK_USED(scene);
        ENOKI_MARK_USED(sensor);
        ENOKI_MARK_USED(diff_scale_factor);
        ENOKI_MARK_USED(pixel_count);
        ENOKI_MARK_USED(sample_count);
        Throw("Not implemented for CUDA arrays.");
    }
}

MTS_VARIANT void
PathLengthOriginIntegrator<Float, Spectrum>::render_sample(const Scene *scene,
                                                   const Sensor *sensor,
                                                   Sampler *sampler,
                                                   ImageBlock *block,
                                                   const Vector2f &pos,
                                                   ScalarFloat diff_scale_factor,
                                                   Mask active) const {
    Vector2f position_sample = pos + sampler->next_2d(active);

    Point2f aperture_sample(.5f);
    if (sensor->needs_aperture_sample())
        aperture_sample = sampler->next_2d(active);

    Float time = sensor->shutter_open();
    if (sensor->shutter_open_time() > 0.f)
        time += sampler->next_1d(active) * sensor->shutter_open_time();

    Float wavelength_sample = sampler->next_1d(active);

    Vector2f adjusted_position =
        (position_sample - sensor->film()->crop_offset()) /
        sensor->film()->crop_size();

    auto [ray, ray_weight] = sensor->sample_ray_differential(
        time, wavelength_sample, adjusted_position, aperture_sample);

    ray.scale_differential(diff_scale_factor);

    const Medium *medium = sensor->medium();
    std::vector<Spectrum> result_spectra;
    std::vector<std::vector<Point3f>> result_interaction_points;
    Mask result_mask;
    std::tie(result_spectra, result_interaction_points, result_mask) = sample_with_length_and_origin(scene, sampler, ray, medium, active);
    
    for(int i=0;i<result_spectra.size();i++)
        result_spectra[i] += ray_weight * result_spectra[i];

    /*UnpolarizedSpectrum spec_u = depolarize(result_spectra[0]);

    Color3f xyz;
    if constexpr (is_monochromatic_v<Spectrum>) {
        xyz = spec_u.x();
    } else if constexpr (is_rgb_v<Spectrum>) {
        xyz = srgb_to_xyz(spec_u, active);
    } else {
        static_assert(is_spectral_v<Spectrum>);
        xyz = spectrum_to_xyz(spec_u, ray.wavelengths, active);
    }

    block->put(position_sample, nullptr, active);*/

    sampler->advance();
}

MTS_VARIANT std::tuple<std::vector<Spectrum>, std::vector<std::vector<typename PathLengthOriginIntegrator<Float, Spectrum>::Point3f>>, typename PathLengthOriginIntegrator<Float, Spectrum>::Mask>
PathLengthOriginIntegrator<Float, Spectrum>::sample_with_length_and_origin(const Scene *scene,
                                                                           Sampler *sampler,
                                                                           const RayDifferential3f &ray_,
                                                                           const Medium * /*medium*/,
                                                                           Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);
    
    // Select ray
    RayDifferential3f ray = ray_;

    // Tracks radiance scaling due to index of refraction changes
    Float eta(1.f);

    // MIS weight for intersected emitters (set by prev. iteration)
    Float emission_weight(1.f);

    Spectrum throughput(1.f), weight_single_ray(0.f); 

    // vector to store distances between all surface interactions
    std::vector<Point3f> ray_interaction_points;

    // return vectors to return all results from all possible emitters
    std::vector<Spectrum> ray_weights;
    std::vector<std::vector<Point3f>> ray_interaction_points_list;

    // DEBUGGING
    // Track weights for each intersection
    /*std::vector<Point3f> origins_sampled_rays;
    std::vector<Vector3f> directions_sampled_rays;
    std::vector<Spectrum> throughput_vector;
    std::vector<std::vector<Spectrum>> throughput_vector_from_all_interactions_list;
    std::vector<Float> emission_weight_vector;
    std::vector<std::vector<Float>> emission_weight_vector_from_all_interactions_list;
    std::vector<Spectrum> emitter_eval_from_all_emissions;*/

    // store ray origin in vector
    ray_interaction_points.push_back(ray.o);
    
    // DEBUGGING
    /*throughput_vector.push_back(throughput);
    emission_weight_vector.push_back(emission_weight);
    origins_sampled_rays.push_back(ray.o);
    directions_sampled_rays.push_back(ray.d);*/

    // ---------------------- First intersection ----------------------

    SurfaceInteraction3f si = scene->ray_intersect(ray, active);
    Mask valid_ray = si.is_valid();
    EmitterPtr emitter = si.emitter(scene);

    // DEBUGGING:
    /*std::cout << "New ray:\n";
    if(none(valid_ray)) {
        std::cout << "ray invalid - ray.o: " << ray.o << "\n";
    }*/

    for (int depth = 1;; ++depth) {

        
        ray_interaction_points.push_back(si.p);
        throughput *= norm(ray_interaction_points[depth]-ray_interaction_points[depth-1])/2/M_PI/si.compute_abs_cos_theta(si.wi);  //Scale throughput to match US derivation (cos_abs_theta sensor is missing, because the ray is already sampled)
    

        // ---------------- Intersection with emitters (ray directly hits emitter) ----------------
        /* TODO: This only works for the CPU implementation -> For the GPU case this will be evaluated in any case -> Resetting of weights does not work */
        if (any_or<true>(neq(emitter, nullptr))) {
            weight_single_ray[active] = emission_weight * throughput * emitter->eval(si, active);
            
            // Add result to output vector
            ray_weights.push_back(weight_single_ray);
            
            // Add interaction points to list
            ray_interaction_points_list.push_back(ray_interaction_points);

            // DEBUGGING
            /*throughput_vector_from_all_interactions_list.push_back(throughput_vector);
            emitter_eval_from_all_emissions.push_back(emitter->eval(si, active));
            std::cout << "Direct intersection with emitter\n";
            std::cout << "Depth: " << depth+1 << "\n";
            std::cout << "Ray origin: " << ray_interaction_points[0] << "\n";
            std::cout << "Last interaction point: " << ray_interaction_points[1] << "\n";
            std::cout << "Emitter position: " << emitter->get_p() << "\n";
            std::cout << "All interaction points: " << ray_interaction_points << "\n";
            std::cout << "Weight: " << weight_single_ray << "\n";
            std::cout << "throughput: " << throughput << "\n";
            std::cout << "emitter->eval(si, active): " << emitter->eval(si, active) << "\n";
            std::cout << "si.shape->sample_position(0.f, Point2f(1.f,1.f),active): " << si.shape->sample_position(0.f, Point2f(1.f,1.f),active) << "\n";
            std::cout << "Ray origins: " << origins_sampled_rays << "\n";
            std::cout << "Ray directions: " << directions_sampled_rays << "\n";*/

            // Clear weight of ray
            weight_single_ray = Spectrum(0.f);                       

            // Set rays inactive that hit the emitter
            active = andnot(active,neq(emitter, nullptr));

        }
                
        active &= si.is_valid();   

        // Russian roulette: try to keep path weights equal to one,
        // while accounting for the solid angle compression at refractive
        // index boundaries. Stop with at least some probability to avoid
        // getting stuck (e.g. due to total internal reflection)
        if (depth > m_rr_depth) {
            Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
            active &= sampler->next_1d(active) < q;
            throughput *= rcp(q);
        }

        // Stop if we've exceeded the number of requested bounces, or
        // if there are no more active lanes. Only do this latter check
        // in GPU mode when the number of requested bounces is infinite
        // since it causes a costly synchronization.
        if ((uint32_t) depth >= (uint32_t) m_max_depth){
            // Add result to output vector
            ray_weights.push_back(Spectrum(0.f));
            
            // Add vector of interaction points to output vector
            ray_interaction_points_list.push_back(ray_interaction_points);
            
            break;
        }
        else if ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active))
            break;

        // Stop if the shape is associated with a sensor (no reflections from sensors permitted)
        active = andnot(active,si.is_sensor());

        // --------------------- Emitter sampling  ---------------------
        // -- Sample emitters in direction of surface interaction & check for ray hit ---

        BSDFContext ctx;
        BSDFPtr bsdf = si.bsdf(ray);
        Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

        if (likely(any_or<true>(active_e))) {
            auto [ds, emitter_val] = scene->sample_emitter_direction(
                si, sampler->next_2d(active_e), true, active_e);
            active_e &= neq(ds.pdf, 0.f);

            // DEBUGGING:
            std::cout << 'ds:' << ds << '\n';

            // Query the BSDF for that emitter-sampled direction
            Vector3f wo = si.to_local(ds.d);
            Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

            // Scale bsdf val according to US match
            bsdf_val *= norm(ds.p-si.p)/2/M_PI/si.compute_abs_cos_theta(wo);

            // Determine density of sampling that same direction using BSDF sampling
            Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);

            Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));                    // mis: multiple importance sampling
            weight_single_ray[active] = mis * throughput * bsdf_val * emitter_val;      // SCALING BY COS_THETA OF THE OUTGOING DIRECTION (BSDF) AND INCOMING DIRECTION (EMITTER), THEIR DISTANCE AND 2*PI MISSING
            

            // Add result to output vector
            ray_weights.push_back(weight_single_ray);
            
            // Add vector of interaction points to output vector
            std::vector<Point3f> ray_interaction_points_complete = ray_interaction_points;
            ray_interaction_points_complete.push_back(ds.p);
            ray_interaction_points_list.push_back(ray_interaction_points_complete);

            // DEBUGGING:
            /*throughput_vector_from_all_interactions_list.push_back(throughput_vector);
            emitter_eval_from_all_emissions.push_back(emitter_val);*/
            std::cout << "Emitter sampling successful\n";
            std::cout << "Depth: " << (depth+1) << "\n";
            /*std::cout << "Ray origin: " << ray_interaction_points[0] << "\n";
            std::cout << "Last interaction point: " << ray_interaction_points_complete[1] << "\n";
            std::cout << "Emitter position: " << ds.p << "\n";
            std::cout << "All interaction points: " << ray_interaction_points_list.back() << "\n";
            std::cout << "Weight: " << weight_single_ray << "\n";
            std::cout << "Ray origins: " << origins_sampled_rays << ", " << si.p << "\n";
            std::cout << "Ray directions: " << directions_sampled_rays << ", " << ds.d <<"\n";
            std::cout << "mis: " << mis <<"\n"; 
            std::cout << "throughput: " << throughput <<"\n";
            std::cout << "bsdf: " << bsdf << "\n";
            std::cout << "bsdf_val: " << bsdf_val <<"\n";
            std::cout << "emitter_val: " << emitter_val <<"\n";
            std::cout << "ds: " << ds << "\n";*/

            // Clear weight of ray
            weight_single_ray = Spectrum(0.f);
            
        }

        // ----------------------- BSDF sampling ----------------------

        // Sample BSDF * cos(theta)
        auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active),
                                           sampler->next_2d(active), active);
        bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

        // DEBUGGING:
        /*if (depth == 1){
           auto cos_theta_i =  si.compute_abs_cos_theta(si.wi);
           auto cos_theta_o =  si.compute_abs_cos_theta(bs.wo);
           std::cout << "First intersection: cos_theta_i: " << cos_theta_i << ', test: ' << abs(dot(si.n,si.wi))<< "\n";
           std::cout << "First intersection: cos_theta_o: " << cos_theta_o << ', test: ' << abs(dot(si.n,bs.wo))<< "\n";
           std::cout << "First intersection: si.n " << si.n << '\n';
           std::cout << "First intersectiom: bs: " << bs << "\n";
        }*/

        throughput = throughput * sqrt(bsdf_val);           // TODO: THIS ONLY WORKS FOR REFLECTIONS. ADJUST the value accordingly
        active &= any(neq(depolarize(throughput), 0.f));
        if (none_or<false>(active))
            break;

        eta *= bs.eta;

        // Intersect the BSDF ray against the scene geometry
        ray = si.spawn_ray(si.to_world(bs.wo));
        SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

        throughput /= si.compute_abs_cos_theta(bs.wo);        // Update factor to scale according to US implementations

        // DEBUGGING
        /*origins_sampled_rays.push_back(ray.o);
        directions_sampled_rays.push_back(ray.d);*/

        // Determine probability of having sampled that same
        // direction using emitter sampling. 
        emitter = si_bsdf.emitter(scene, active);
        DirectionSample3f ds(si_bsdf, si);
        ds.object = emitter;

        if (any_or<true>(neq(emitter, nullptr))) {
            Float emitter_pdf =
                select(neq(emitter, nullptr) && !has_flag(bs.sampled_type, BSDFFlags::Delta),
                       scene->pdf_emitter_direction(si, ds),
                       0.f);

            emission_weight = mis_weight(bs.pdf, emitter_pdf);

            // DEBUGGING:
            /*std::cout << "Same direction already sampled by emitter sampling:\n";
            std::cout << "emission_weight: " << emission_weight << "\n";*/
        }

        si = std::move(si_bsdf);
        //ray_interaction_points.push_back(si.p);
        
        // DEBUGGING:
        //throughput_vector.push_back(throughput);
    }

    

    return {ray_weights, ray_interaction_points_list, valid_ray};
}

// -----------------------------------------------------------------------------

MTS_IMPLEMENT_CLASS_VARIANT(Integrator, Object, "integrator")
MTS_IMPLEMENT_CLASS_VARIANT(SamplingIntegrator, Integrator)
MTS_IMPLEMENT_CLASS_VARIANT(MonteCarloIntegrator, SamplingIntegrator)
MTS_IMPLEMENT_CLASS_VARIANT(PathLengthOriginIntegrator, MonteCarloIntegrator)

MTS_INSTANTIATE_CLASS(Integrator)
MTS_INSTANTIATE_CLASS(SamplingIntegrator)
MTS_INSTANTIATE_CLASS(MonteCarloIntegrator)
MTS_INSTANTIATE_CLASS(PathLengthOriginIntegrator)

NAMESPACE_END(mitsuba)
