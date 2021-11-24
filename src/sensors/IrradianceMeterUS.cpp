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

MTS_VARIANT class IrradianceMeterUS final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_film, m_world_transform, m_shape)
    MTS_IMPORT_TYPES(Shape, Texture)

    IrradianceMeterUS(const Properties &props) : Base(props) {
        if (props.has_property("to_world"))
            Throw("Found a 'to_world' transformation -- this is not allowed. "
                  "The irradiance meter inherits this transformation from its parent "
                  "shape.");

        if (m_film->size() != ScalarPoint2i(1, 1))
            Throw("This sensor only supports films of size 1x1 Pixels!");
        
        // Assign bounds of cuboid to sample rays from
        if (props.has_property("r_focus_elevational"))
            m_r_focus_elevational = props.float_("r_focus_elevational");
        else
            Throw("This sensor requires the radial distance to the focal point in elevation (r_focus_elevational, in m)!");
        
        if (props.has_property("r_focus_in_plane"))
            m_r_focus_in_plane = props.float_("r_focus_in_plane");
        else
            Throw("This sensor requires the radial distance to the focal point in elevation (r_focus_in_plane, in m)!");
        
        if (props.has_property("r_max_bound_SPMR"))
            m_r_max_bound_SPMR = props.float_("r_max_bound_SPMR");
        else
            Throw("This sensor requires the radial distance corresponding to the largest azimuthal angle of the SPMR -3dB line (r_max_bound_SPMR, in m)!");
       
        if (props.has_property("phi_max_bound_SPMR")) {
            m_phi_max_bound_SPMR = deg_to_rad(props.float_("phi_max_bound_SPMR"));
        } else
            Throw("This sensor requires the largest in-plane angle of the SPMR -3dB line (phi_max_bound_SPMR, in degree)!");

        if (props.has_property("y_max_bound_SPMR"))
            m_y_max_bound_SPMR = props.float_("y_max_bound_SPMR");
        else
            Throw("This sensor requires an out-of-plane elevation bound of the -3dB line at the smallest(y_max_bound_SPMR, in m) for the bounding box to sample the ray direction!");
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
        //Vector3f dir_ray = sample_dir_from_SIR(sample3, ps);
        Vector3f dir_ray = warp::square_to_cosine_hemisphere(sample3);
        
        // 3. Sample spectrum
        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);

        // TODO: Generalize scaling of weight for different transducer shapes

        return std::make_pair(
            RayDifferential3f(ps.p, Frame3f(ps.n).to_world(dir_ray), time, wavelengths),
            unpolarized<Spectrum>(wav_weight) * math::Pi<ScalarFloat> * m_r_focus_elevational       // pi scaling: cos-weighted hemisphere direction, radius scaling: area element of cylindrical transducer shape
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
        oss << "IrradianceMeterUS[" << std::endl
            << "  shape = " << m_shape << "," << std::endl
            << "  film = " << m_film << "," << std::endl
            << "  m_r_focus_elevational = " << m_r_focus_elevational << "m," << std::endl
            << "  m_r_focus_in_plane = " << m_r_focus_in_plane << "m," << std::endl
            << "  m_r_max_bound_SPMR = " << m_r_max_bound_SPMR << "m," << std::endl
            << "  m_phi_max_bound_SPMR = " << rad_to_deg(m_phi_max_bound_SPMR) << "deg," << std::endl
            << "  m_y_max_bound_SPMR = " << m_y_max_bound_SPMR << "m" << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

protected:
    ScalarFloat m_r_focus_elevational;      // Radial distance to focus point for elevational focusing
    ScalarFloat m_r_focus_in_plane;         // Radial distance to focus point for in-plane arrangement
    ScalarFloat m_r_max_bound_SPMR;         // Radial distance at which m_phi_max_bound_SPMR is applicable
    ScalarFloat m_phi_max_bound_SPMR;       // Bound of SPMR energy (-3dB line) in azimuthal-dimension (lateral, radians) -> std of in-plane angle of rays
    ScalarFloat m_y_max_bound_SPMR;         // Bound of transducer sensitivity in y-dimension (out-of-plane) -> std of out-of-plane direction of rays
    ref<Texture> m_radiance;

    Vector3f sample_dir_from_SIR(const Point2f &sample3, PositionSample3f ps) const {
        // Sample elevation & in-plane angle
        //Float y_samp = 2*m_y_max_bound*sample3.y() - m_y_max_bound;
        //Float phi_samp = 2*m_phi_max_bound*sample3.x() - m_phi_max_bound;
        
        // Sample elevation & in-plane angle from Normal distribution (mapped with Box-Muller transform)
        Float y_samp = m_y_max_bound_SPMR*sqrt(-2.f*log(sample3.x()))*cos(2.f*math::Pi<Float>*sample3.y());
        Float phi_samp = m_phi_max_bound_SPMR*sqrt(-2.f*log(sample3.x()))*sin(2.f*math::Pi<Float>*sample3.y());

        // Compute the direction of the ray & normalize
        Vector3f dir_ray = Vector3f(m_r_max_bound_SPMR*sin(phi_samp),y_samp-ps.p.y(),m_r_max_bound_SPMR*cos(phi_samp));
        dir_ray /= norm(dir_ray);

        return dir_ray;
    }

};


MTS_IMPLEMENT_CLASS_VARIANT(IrradianceMeterUS, Sensor)
MTS_EXPORT_PLUGIN(IrradianceMeterUS, "IrradianceMeterUS");
NAMESPACE_END(mitsuba)
