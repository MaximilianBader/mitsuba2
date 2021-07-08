#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-spot:

Spot light source (:monosp:`spot`)
------------------------------------

.. pluginparameters::

 * - intensity
   - |spectrum|
   - Specifies the maximum radiant intensity at the center in units of power per unit steradian. (Default: 1).
     This cannot be spatially varying (e.g. have bitmap as type).

 * - cutoff_angle
   - |float|
   - Cutoff angle, beyond which the spot light is completely black (Default: 20 degrees)

 * - beam_width
   - |float|
   - Subtended angle of the central beam portion (Default: :math:`cutoff_angle \times 3/4`)

 * - texture
   - |texture|
   - An optional texture to be projected along the spot light. This must be spatially varying (e.g. have bitmap as type).

 * - to_world
   - |transform|
   - Specifies an optional emitter-to-world transformation.  (Default: none, i.e. emitter space = world space)

This plugin provides a spot light with a linear falloff. In its local coordinate system, the spot light is
positioned at the origin and points along the positive Z direction. It can be conveniently reoriented
using the lookat tag, e.g.:

.. code-block:: xml
    :name: spot-light

    <emitter type="spot">
        <transform name="to_world">
            <!-- Orient the light so that points from (1, 1, 1) towards (1, 2, 1) -->
            <lookat origin="1, 1, 1" target="1, 2, 1"/>
        </transform>
    </emitter>

The intensity linearly ramps up from cutoff_angle to beam_width (both specified in degrees),
after which it remains at the maximum value. A projection texture may optionally be supplied.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/emitter_spot_no_texture.jpg
   :caption: Two spot lights with different colors and no texture specified.
.. subfigure:: ../../resources/data/docs/images/render/emitter_spot_texture.jpg
   :caption: A spot light with a texture specified.
.. subfigend::
   :label: fig-spot-light

 */

template <typename Float, typename Spectrum>
class UltrasoundEmitter final : public Emitter<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Emitter, m_flags, m_medium, m_world_transform)
    MTS_IMPORT_TYPES(Scene, Texture)

    UltrasoundEmitter(const Properties &props) : Base(props) {
        m_flags = +EmitterFlags::DeltaPosition;
        m_intensity = props.texture<Texture>("intensity", Texture::D65(1.f));
        m_texture = props.texture<Texture>("texture", Texture::D65(1.f));

        if (m_intensity->is_spatially_varying())
            Throw("The parameter 'intensity' cannot be spatially varying (e.g. bitmap type)!");

        if (props.has_property("texture")) {
            if (!m_texture->is_spatially_varying())
                Throw("The parameter 'texture' must be spatially varying (e.g. bitmap type)!");
            m_flags |= +EmitterFlags::SpatiallyVarying;
        }

        // Assign bounds of cuboid to sample rays from
        if (props.has_property("r_min_bound"))
            m_r_min_bound = props.float_("r_min_bound");
        else
            Throw("This emitter requires an axial bound (r_min_bound, in m) for the bounding box to sample the ray direction!");
       
        if (props.has_property("phi_max_bound")) {
            m_phi_max_bound = deg_to_rad(props.float_("phi_max_bound"));
        } else
            Throw("This emitter requires a lateral bound (phi_max_bound, in degree) for the bounding box to sample the ray direction!");

        if (props.has_property("y_max_bound"))
            m_y_max_bound = props.float_("y_max_bound");
        else
            Throw("This emitter requires an out-of-plane elevation bound (y_max_bound, in m) for the bounding box to sample the ray direction!");
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &spatial_sample,
                                          const Point2f & /*dir_sample*/,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample directional component
        const Transform4f &trafo = m_world_transform->eval(time, active);
        //Vector3f local_dir = warp::square_to_uniform_cone(spatial_sample, (Float)m_cos_cutoff_angle);
        Vector3f local_dir = square_to_polar_bounding_box_surface(spatial_sample);
        //Float pdf_dir = warp::square_to_uniform_cone_pdf(local_dir, (Float)m_cos_cutoff_angle);

        // 2. Sample spectrum
        auto [wavelengths, spec_weight] = m_intensity->sample_spectrum(
            zero<SurfaceInteraction3f>(),
            math::sample_shifted<Wavelength>(wavelength_sample), active);

        //UnpolarizedSpectrum falloff_spec = falloff_curve(local_dir, wavelengths, active);

        return { Ray3f(trafo.translation(), trafo * local_dir, time, wavelengths),
                unpolarized<Spectrum>(spec_weight) };
                //unpolarized<Spectrum>(falloff_spec / pdf_dir) };
    }

    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f &/*sample*/,
                                                            Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        Transform4f trafo = m_world_transform->eval(it.time, active);

        DirectionSample3f ds;
        ds.p        = trafo.translation();
        ds.n        = 0.f;
        ds.uv       = 0.f;
        ds.pdf      = 1.0f;
        ds.time     = it.time;
        ds.delta    = true;
        ds.object   = this;
        ds.d        = ds.p - it.p;
        ds.dist     = norm(ds.d);
        Float inv_dist = rcp(ds.dist);
        ds.d        *= inv_dist;
        Vector3f local_d = trafo.inverse() * -ds.d;
        //UnpolarizedSpectrum falloff_spec = falloff_curve(local_d, it.wavelengths, active);

        auto weight = select(abs(acos(local_d.z())) <= m_phi_max_bound, unpolarized<Spectrum>(1.f), unpolarized<Spectrum>(0.f));
        weight = select(abs(local_d.y()) <= m_y_max_bound, weight, unpolarized<Spectrum>(0.f));

        return { ds, weight};//unpolarized<Spectrum>(weight) };
    }

    Float pdf_direction(const Interaction3f &, const DirectionSample3f &, Mask) const override {
        return 0.f;
    }

    Spectrum eval(const SurfaceInteraction3f &, Mask) const override { return 0.f; }

    ScalarBoundingBox3f bbox() const override {
        return m_world_transform->translation_bounds();
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("intensity", m_intensity.get());
        callback->put_object("texture", m_texture.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SpotLight[" << std::endl
            << "  world_transform = " << string::indent(m_world_transform) << "," << std::endl
            << "  intensity = " << m_intensity << "," << std::endl
            << "  m_r_min_bound = " << m_r_min_bound << "," << std::endl
            << "  m_phi_max_bound = " << m_phi_max_bound << "," << std::endl
            << "  m_y_max_bound = " << m_y_max_bound << "," << std::endl
            << "  texture = " << (m_texture ? string::indent(m_texture) : "")
                        << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_intensity;
    ref<Texture> m_texture;
    ScalarFloat m_r_min_bound;         // Bound of Fov in radial-dimension (axial) -> lateral bound for rays to sample
    ScalarFloat m_phi_max_bound;       // Bound of Fov in azimuthal-dimension (lateral, radians) -> axial bound for rays to sample
    ScalarFloat m_y_max_bound;         // Bound of transducer sensitivity in y-dimension (out-of-plane) -> elevational bound for rays_to_sample

    Vector3f square_to_polar_bounding_box_surface(const Point2f &point_on_square) const {
        Float y_samp = 2*m_y_max_bound*point_on_square.y() - m_y_max_bound;
        Float r_in_plane = safe_sqrt(1.f-y_samp*y_samp);
        Float phi_samp = 2*m_phi_max_bound*point_on_square.x() - m_phi_max_bound;
        
        return {r_in_plane*sin(phi_samp),y_samp,r_in_plane*cos(phi_samp)};
    }
};


MTS_IMPLEMENT_CLASS_VARIANT(UltrasoundEmitter, Emitter)
MTS_EXPORT_PLUGIN(UltrasoundEmitter, "Ultrasound transducer emitter")
NAMESPACE_END(mitsuba)
