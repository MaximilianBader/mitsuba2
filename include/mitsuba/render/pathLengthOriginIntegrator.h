#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/tls.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/imageblock.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/integrator.h>


NAMESPACE_BEGIN(mitsuba)

/*
 * \brief This integrator implements a basic path tracer including the path length and the last interaction point
 */
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER PathLengthOriginIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, 
                    /* member variables*/ m_max_depth, m_rr_depth, m_stop, m_samples_per_pass,m_render_timer,m_timeout,m_block_size, 
                    /* member functions*/ should_stop, aov_names)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr, Sensor, Film, ImageBlock)

    /// Create an integrator
    PathLengthOriginIntegrator(const Properties &props);

    // general render fct
    bool render_with_length(Scene *scene, Sensor *sensor);

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
                                                                                         Mask active = true) const;

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */ ,
                                     Float * /* aovs */ ,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);
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

protected:

    void render_block(const Scene *scene,
                      const Sensor *sensor,
                      Sampler *sampler,
                      ImageBlock *block,
                      size_t sample_count,
                      size_t block_id) const;

    void render_sample(const Scene *scene,
                       const Sensor *sensor,
                       Sampler *sampler,
                       ImageBlock *block,
                       const Vector2f &pos,
                       ScalarFloat diff_scale_factor,
                       Mask active = true) const;

};

MTS_EXTERN_CLASS_RENDER(PathLengthOriginIntegrator)

NAMESPACE_END(mitsuba)
