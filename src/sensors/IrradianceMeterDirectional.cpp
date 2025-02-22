#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/sensor.h>
//#include <math.h>

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
        
        // Assign bounds of cuboid to sample rays from
        if (props.has_property("r_min_bound"))
            m_r_min_bound = props.float_("r_min_bound");
        else
            Throw("This sensor requires an axial bound (r_min_bound, in m) for the bounding box to sample the ray direction!");
       
        if (props.has_property("phi_max_bound")) {
            m_phi_max_bound = deg_to_rad(props.float_("phi_max_bound"));
        } else
            Throw("This sensor requires a lateral bound (phi_max_bound, in degree) for the bounding box to sample the ray direction!");

        if (props.has_property("y_max_bound"))
            m_y_max_bound = props.float_("y_max_bound");
        else
            Throw("This sensor requires an out-of-plane elevation bound (y_max_bound, in m) for the bounding box to sample the ray direction!");
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
        //Vector3f dir_ray = warp::square_to_cosine_hemisphere(sample3);
        Vector3f dir_ray = sample_dir_from_SIR(sample3, ps);
        //Vector3f dir_ray = square_to_polar_bounding_box_surface(sample3);

        // 3. Sample spectrum
        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);

        return std::make_pair(
            RayDifferential3f(ps.p, Frame3f(ps.n).to_world(dir_ray), time, wavelengths),
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
            << "  r_box_bound = " << m_r_min_bound << "," << std::endl
            << "  phi_box_bound = " << m_phi_max_bound << "," << std::endl
            << "  y_box_bound = " << m_y_max_bound << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

protected:
    ScalarFloat m_r_min_bound;         // Bound of Fov in radial-dimension (axial) -> lateral bound for rays to sample
    ScalarFloat m_phi_max_bound;       // Bound of Fov in azimuthal-dimension (lateral, radians) -> axial bound for rays to sample
    ScalarFloat m_y_max_bound;         // Bound of transducer sensitivity in y-dimension (out-of-plane) -> elevational bound for rays_to_sample

    /*Vector3f sample_dir_from_FoV(const Point1f &sample1, const Point2f &sample3, PositionSample3f ps) const {
        // Sample elevation & in-plane angle
        Float x_samp = 2*0.02;
        Float y_samp = 2*m_y_max_bound*sample3.y() - m_y_max_bound;
        Float phi_samp = 2*m_phi_max_bound*sample3.x() - m_phi_max_bound;
        
        // Compute the direction of the ray & normalize
        Vector3f dir_ray = Vector3f(m_r_min_bound*sin(phi_samp),y_samp-ps.p.y(),m_r_min_bound*cos(phi_samp));
        dir_ray /= norm(dir_ray);

        return dir_ray;
    }*/

    Vector3f sample_dir_from_SIR(const Point2f &sample3, PositionSample3f ps) const {
        // Sample elevation & in-plane angle
        //Float y_samp = 2*m_y_max_bound*sample3.y() - m_y_max_bound;
        //Float phi_samp = 2*m_phi_max_bound*sample3.x() - m_phi_max_bound;
        
        // Sample elevation & in-plane angle from Normal distribution (mapped with Box-Muller transform)
        Float y_samp = m_y_max_bound*sqrt(-2*log(sample3.x()))*cos(2*math::Pi<Float>*sample3.y());
        Float phi_samp = m_phi_max_bound*sqrt(-2*log(sample3.x()))*sin(2*math::Pi<Float>*sample3.y());

        // Compute the direction of the ray & normalize
        Vector3f dir_ray = Vector3f(m_r_min_bound*sin(phi_samp),y_samp-ps.p.y(),m_r_min_bound*cos(phi_samp));
        dir_ray /= norm(dir_ray);

        return dir_ray;
    }

    Vector3f square_to_polar_bounding_box_surface(const Point2f &point_on_square) const {
        Float y_samp = 2*m_y_max_bound*point_on_square.y() - m_y_max_bound;
        Float r_in_plane = safe_sqrt(1.f-y_samp*y_samp);
        Float phi_samp = 2*m_phi_max_bound*point_on_square.x() - m_phi_max_bound;
        
        return {r_in_plane*sin(phi_samp),y_samp,r_in_plane*cos(phi_samp)};
    }

};


MTS_IMPLEMENT_CLASS_VARIANT(IrradianceMeterDirectional, Sensor)
MTS_EXPORT_PLUGIN(IrradianceMeterDirectional, "IrradianceMeterBoundingBox");
NAMESPACE_END(mitsuba)
