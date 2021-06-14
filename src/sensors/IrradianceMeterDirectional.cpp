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
            <float name="x_max_bound" value="0.02"/>
            <float name="y_max_bound" value="0.005"/>
            <float name="z_max_bound" value="0.02"/>
        </sensor>
    </shape>
*/

MTS_VARIANT class IrradianceMeterDirectional final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_film, m_world_transform, m_shape)
    MTS_IMPORT_TYPES(Shape)

    IrradianceMeterDirectional(const Properties &props) : Base(props) {
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
        
        // Assign bounds of cuboid to sample rays from
        if (props.has_property("x_max_bound"))
            m_x_max_bound = props.float_("x_max_bound");
        else
            Throw("This sensor requires a lateral bound (x_max_bound, in m) for the bounding box to sample the ray direction!");
       
        if (props.has_property("y_max_bound"))
            m_y_max_bound = props.float_("y_max_bound");
        else
            Throw("This sensor requires an out-of-plane elevation bound (y_max_bound, in m) for the bounding box to sample the ray direction!");
         
        if (props.has_property("z_max_bound"))
            m_z_max_bound = props.float_("z_max_bound");
        else
            Throw("This sensor requires an axial bound (z_max_bound, in m) for the bounding box to sample the ray direction!");
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
        Point3f point_fov = sample_point_from_FoV(wavelength_sample,sample3);
        Vector3f dir_ray = point_fov-ps.p;
        dir_ray /= norm(dir_ray);

        // 3. Sample spectrum
        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);

        return std::make_pair(
            RayDifferential3f(ps.p, dir_ray, time, wavelengths),
            unpolarized<Spectrum>(wav_weight) * math::Pi<ScalarFloat>
        );
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        return std::make_pair(m_shape->sample_direction(it, sample, active), math::Pi<ScalarFloat>);
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
        oss << "IrradianceMeterDirectional[" << std::endl
            << "  shape = " << m_shape << "," << std::endl
            << "  film = " << m_film << "," << std::endl
            << "  x_box_bound = " << m_x_max_bound << "," << std::endl
            << "  y_box_bound = " << m_y_max_bound << "," << std::endl
            << "  z_box_bound = " << m_z_max_bound << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

protected:
    Float m_x_max_bound;         // Bound of Fov in x-dimension (lateral) -> lateral bound for rays to sample
    Float m_y_max_bound;         // Bound of transducer sensitivity in y-dimension (out-of-plane) -> elevational bound for rays_to_sample
    Float m_z_max_bound;         // Bound of Fov in z-dimension (axial) -> axial bound for rays to sample

    Point3f sample_point_from_FoV(const Float &sample1,const Point2f &sample3) const {
        return Point3f(2*m_x_max_bound*sample3[0]-m_x_max_bound,
                        2*m_y_max_bound*sample1-m_y_max_bound,
                        2*m_z_max_bound*sample3[1]-m_z_max_bound);
    }
};

// =============================================================================
// Helper functions
// =============================================================================
/*template <typename Value>
MTS_INLINE Vector<Value, 3> square_to_polar_bounding_box_surface(const Point<Value, 2> &sample3){
    
    
    return {0.0f,0.0f,0.0f};
}*/



MTS_IMPLEMENT_CLASS_VARIANT(IrradianceMeterDirectional, Sensor)
MTS_EXPORT_PLUGIN(IrradianceMeterDirectional, "IrradianceMeterBoundingBox");
NAMESPACE_END(mitsuba)
