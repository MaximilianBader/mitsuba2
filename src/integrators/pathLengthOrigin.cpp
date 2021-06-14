#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path_length_origin:

Path tracer (:monosp:`path_length_origin`)
-------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). A value of 1 will only render directly visible light sources. 2 will lead
     to single-bounce (direct-only) illumination, and so on. (Default: -1)
 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start to use the
     *russian roulette* path termination criterion. (Default: 5)
 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This integrator implements a basic path tracer including the path length and the last interaction point

 */

/*
 * \brief This integrator implements a basic path tracer including the path length and the last interaction point
 */
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER PathLengthOriginIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    /// Create an integrator
    PathLengthOriginIntegrator(const Properties &props) : Base(props) {}

    /**
     * \brief Sample the incident radiance along a ray including tracking of wavelength.
     *        This function replaces the basic sample function
     *
     * \param scene
     *    The underlying scene in which the radiance function should be sampled
     *
     * \param sampler
     *    A source of (pseudo-/quasi-) random numbers
     *
     * \param ray
     *    A ray, optionally with differentials
     *
     * \param medium
     *    If the ray is inside a medium, this parameter holds a pointer to that
     *    medium
     *
     * \param active
     *    A mask that indicates which SIMD lanes are active
     *
     * \return
     *    A tuple containing a 
     *      - spectrum 'result' weight/strength associated with ray
     *      - Point3f 'last_interaction_point'
     *      - std::vector<Float> 'covered_distances'
     *      - mask 'valid_ray' specifying whether a surface or medium interaction was sampled.
     *        False mask entries indicate that the ray "escaped" the scene, in which case the 
     *        the returned spectrum contains the contribution of environment maps, if present.
     *        The mask can be used to estimate a suitable alpha channel of a rendered image.
     */
    std::tuple<std::vector<Spectrum>, Point3f, std::vector<std::vector<Float>>, Mask>sample_with_length_and_origin(const Scene *scene,
                                                                                         Sampler * sampler,
                                                                                         const RayDifferential3f &ray_,
                                                                                         const Medium *medium = nullptr,
                                                                                         Mask active = true) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);
    
    // Select ray
    RayDifferential3f ray = ray_;

    // Tracks radiance scaling due to index of refraction changes
    Float eta(1.f);

    // MIS weight for intersected emitters (set by prev. iteration)
    Float emission_weight(1.f);

    Spectrum throughput(1.f), result(0.f), current_result(0.f);

    // vector to store distances between all surface interactions
    std::vector<Float> covered_distances, current_covered_distances;

    // return vectors to return all results from all possible emitters
    std::vector<Spectrum> result_from_all_interactions;
    std::vector<std::vector<Float>> covered_distances_from_all_interactions;

    // ---------------------- First intersection ----------------------

    SurfaceInteraction3f si = scene->ray_intersect(ray, active);
    Point3f last_interaction_point = si.p;
    Mask valid_ray = si.is_valid();
    EmitterPtr emitter = si.emitter(scene);

    for (int depth = 1;; ++depth) {

        // ---------------- Intersection with emitters ----------------

        if (any_or<true>(neq(emitter, nullptr)))
        {
            result[active] += emission_weight * throughput * emitter->eval(si, active);
            std::cout << "Result added in first if statement.\n";
            
            // Add result to output vector
            current_result = result;
            masked(current_result,active) += emission_weight * throughput * emitter->eval(si, active);
            result_from_all_interactions.push_back(current_result);
            
            // Add covered distance to output vector
            current_covered_distances = covered_distances;
            current_covered_distances.push_back(norm(si.p-emitter->get_p()));
            covered_distances_from_all_interactions.push_back(current_covered_distances);
        }
                
        active &= si.is_valid();   

        // Push back travelling distance if there is an active lane
        if (any(active))
            covered_distances.push_back(si.t);

        /* Russian roulette: try to keep path weights equal to one,
           while accounting for the solid angle compression at refractive
           index boundaries. Stop with at least some probability to avoid
           getting stuck (e.g. due to total internal reflection) */
        if (depth > m_rr_depth) {
            Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
            active &= sampler->next_1d(active) < q;
            throughput *= rcp(q);
        }

        // Stop if we've exceeded the number of requested bounces, or
        // if there are no more active lanes. Only do this latter check
        // in GPU mode when the number of requested bounces is infinite
        // since it causes a costly synchronization.
        if ((uint32_t) depth >= (uint32_t) m_max_depth ||
           ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active)))
                break;
            

        // --------------------- Emitter sampling ---------------------

        BSDFContext ctx;
        BSDFPtr bsdf = si.bsdf(ray);
        Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

        if (likely(any_or<true>(active_e))) {
            auto [ds, emitter_val] = scene->sample_emitter_direction(
                si, sampler->next_2d(active_e), true, active_e);
            active_e &= neq(ds.pdf, 0.f);

            // Query the BSDF for that emitter-sampled direction
            Vector3f wo = si.to_local(ds.d);
            Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

            // Determine density of sampling that same direction using BSDF sampling
            Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);

            Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
            result[active_e] += mis * throughput * bsdf_val * emitter_val;

            // Add covered distance from emitter to current position
            // Add result to output vector
            current_result = result;
            masked(current_result,active) += mis * throughput * bsdf_val * emitter_val;
            result_from_all_interactions.push_back(current_result);
            
            // Add covered distance to output vector
            current_covered_distances = covered_distances;
            current_covered_distances.push_back(ds.dist);
            covered_distances_from_all_interactions.push_back(current_covered_distances);
        }

        // ----------------------- BSDF sampling ----------------------

        // Sample BSDF * cos(theta)
        auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active),
                                           sampler->next_2d(active), active);
        bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

        throughput = throughput * bsdf_val;
        active &= any(neq(depolarize(throughput), 0.f));
        if (none_or<false>(active))
            break;

        eta *= bs.eta;

        // Intersect the BSDF ray against the scene geometry
        ray = si.spawn_ray(si.to_world(bs.wo));
        SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

        /* Determine probability of having sampled that same
           direction using emitter sampling. */
        emitter = si_bsdf.emitter(scene, active);
        DirectionSample3f ds(si_bsdf, si);
        ds.object = emitter;

        if (any_or<true>(neq(emitter, nullptr))) {
            Float emitter_pdf =
                select(neq(emitter, nullptr) && !has_flag(bs.sampled_type, BSDFFlags::Delta),
                       scene->pdf_emitter_direction(si, ds),
                       0.f);

            emission_weight = mis_weight(bs.pdf, emitter_pdf);
        }

        si = std::move(si_bsdf);
    }

    //return {result , last_interaction_point, covered_distances, valid_ray};
    return {result_from_all_interactions, last_interaction_point, covered_distances_from_all_interactions, valid_ray};
}

    std::pair<Spectrum, Mask> sample(const Scene *,Sampler *,const RayDifferential3f &,const Medium * ,Float * ,Mask ) const override {
        Throw("PathLengthOriginIntegrator:: sample() has been replaced by sample_with_length_and_origin!");
    }

    std::string to_string() const override {
        return tfm::format("PathLengthOriginIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i\n"
            "]", m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    ///  destructor
    ~PathLengthOriginIntegrator() {}

    MTS_DECLARE_CLASS()

};

MTS_IMPLEMENT_CLASS_VARIANT(PathLengthOriginIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PathLengthOriginIntegrator, "Path Tracer integrator including path length and last point of interaction");
NAMESPACE_END(mitsuba)
