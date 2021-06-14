#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-irradiancemeter:

Irradiance meter (:monosp:`irradiancemeter`)
--------------------------------------------

.. pluginparameters::

 * - none

This sensor plugin implements an irradiance meter, which measures
the incident power per unit area over a shape which it is attached to.
This sensor is used with films of 1 by 1 pixels.

If the irradiance meter is attached to a mesh-type shape, it will measure the
irradiance over all triangles in the mesh.

This sensor is not instantiated on its own but must be defined as a child
object to a shape in a scene. To create an irradiance meter,
simply instantiate the desired sensor shape and specify an
:monosp:`irradiancemeter` instance as its child:

.. code-block:: xml
    :name: sphere-meter

    <shape type="sphere">
        <sensor type="irradiancemeter">
            <!-- film -->
        </sensor>
    </shape>
*/

MTS_VARIANT class IrradianceMeter2D final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_film, m_world_transform, m_shape)
    MTS_IMPORT_TYPES(Shape)

    IrradianceMeter2D(const Properties &props) : Base(props) {
        if (props.has_property("to_world"))
            Throw("Found a 'to_world' transformation -- this is not allowed. "
                  "The irradiance meter inherits this transformation from its parent "
                  "shape.");

        if (m_film->size() != ScalarPoint2i(1, 1))
            Throw("This sensor only supports films of size 1x1 Pixels!");

        if (m_film->reconstruction_filter()->radius() >
            0.5f + math::RayEpsilon<Float>)
            Log(Warn, "This sensor should only be used with a reconstruction filter"
               "of radius 0.5 or lower(e.g. default box)");
    }

    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample,
                            const Point2f & sample2,
                            const Point2f & sample3,
                            Mask active) const override {

        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample spatial component
        PositionSample3f ps = m_shape->sample_position(time, sample2, active);

        // 2. Sample directional component
        Vector3f local = warp::square_to_cosine_hemisphere(sample3);

        // 3. Sample spectrum
        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);

        Vector3f d_2D = Frame3f(ps.n).to_world(local);
        d_2D[1] = Float32(0.f);

        return std::make_pair(
            RayDifferential3f(ps.p, d_2D, time, wavelengths),
            unpolarized<Spectrum>(wav_weight) * math::Pi<ScalarFloat>
        );
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        DirectionSample3f ds = m_shape->sample_direction(it, sample, active);
        ds.d[1] = Float(0.f);

        Float dist_squared = squared_norm(ds.d);
        ds.dist = sqrt(dist_squared);
        ds.d /= ds.dist;

        Float dp = abs_dot(ds.d, ds.n);
        ds.pdf *= select(neq(dp, 0.f), dist_squared / dp, 0.f);
        
        return std::make_pair(ds, math::Pi<ScalarFloat>);
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        return m_shape->pdf_direction(it, ds, active);
    }

    Spectrum eval(const SurfaceInteraction3f &/*si*/, Mask /*active*/) const override {
        return math::Pi<ScalarFloat> / m_shape->surface_area();
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "IrradianceMeter2D[" << std::endl
            << "  shape = " << m_shape << "," << std::endl
            << "  film = " << m_film << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(IrradianceMeter2D, Sensor)
MTS_EXPORT_PLUGIN(IrradianceMeter2D, "IrradianceMeter2D");
NAMESPACE_END(mitsuba)
