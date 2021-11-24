import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
import pickle
import mpl_toolkits.mplot3d.art3d as art3d 
import enoki as ek
import mitsuba
from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f, Point3f
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.render import Scene

# --- VISUALIZING SCENE ---
def visualize_scene(scene,image_path,axis_limits = [[-1,1],[-1,1],[-1,1]]):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot all shapes of the scene
    plot_scene_in_ax(scene,ax)
    
    # add axis labels & save the figure
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_zlim(axis_limits[2])
    ax.set_box_aspect(aspect=(1,(axis_limits[1][1]-axis_limits[1][0])/(axis_limits[0][1]-axis_limits[0][0]),(axis_limits[2][1]-axis_limits[2][0])/(axis_limits[0][1]-axis_limits[0][0])))
    image_file_name = image_path / 'scene_visualization.jpeg'
    plt.savefig(image_file_name.resolve(),dpi=600)
    pickle.dump(fig, open((image_path / 'scene_visualization.fig.pickle').resolve(), 'wb'))

    # Loading of pickle result
    # import pickle
    # import matplotlib.pyplot as plt
    # figx = pickle.load(open('E:/Max/02-projects/202105-Ultrasound_mb/02-simulations_ray_tracing/20210909-US_modeling_ray_tracing_transmission_membrane/scene_ray_visualization.fig.pickle','rb'))
    # figx.show()
    # data = figx.axes[0].lines[0].get_data()

# --- VISUALIZING SCENE AND RAY PATHS ---
def visualize_scene_ray_paths(scene,weights_all_rays,interaction_points_all_rays,image_path,axis_limits = [[-1,1],[-1,1],[-1,1]]):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot all shapes of the scene
    plot_scene_in_ax(scene,ax)

    # plot the rays
    for index_ray in np.arange(start=0,stop=len(interaction_points_all_rays)):
        interaction_points_single_ray = interaction_points_all_rays[index_ray]
        if weights_all_rays[index_ray] != 0:
            ray_color = 'black'
            ray_label = "ray (weight=0)"
        else:
            ray_color = 'red'
            ray_label = "ray (weight != 0)"
        
        x_values = np.empty((len(interaction_points_single_ray),))
        y_values = np.empty((len(interaction_points_single_ray),))
        z_values = np.empty((len(interaction_points_single_ray),))
        for index_point in np.arange(start=0,stop=len(interaction_points_single_ray)):
            x_values[index_point] = interaction_points_single_ray[index_point].x
            y_values[index_point] = interaction_points_single_ray[index_point].y
            z_values[index_point] = interaction_points_single_ray[index_point].z
        ax.plot(x_values,y_values,z_values,label=ray_label,color=ray_color)
        ax.scatter3D(x_values,y_values,z_values,c=ray_color)
    
    # add axis labels & save the figure
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_zlim(axis_limits[2])
    ax.set_box_aspect(aspect=(1,(axis_limits[1][1]-axis_limits[1][0])/(axis_limits[0][1]-axis_limits[0][0]),(axis_limits[2][1]-axis_limits[2][0])/(axis_limits[0][1]-axis_limits[0][0])))
    image_file_name = image_path / 'scene_ray_path_visualization.jpeg' #scene_ray_path_visualization
    plt.savefig(image_file_name.resolve(),dpi=600)
    pickle.dump(fig, open((image_path / 'scene_ray_path_visualization.fig.pickle').resolve(), 'wb'))
    
    # Loading of pickle result
    # import pickle
    # import matplotlib.pyplot as plt
    # figx = pickle.load(open('E:/Max/02-projects/202105-Ultrasound_mb/02-simulations_ray_tracing/20210909-US_modeling_ray_tracing_transmission_membrane/scene_ray_visualization.fig.pickle','rb'))
    # figx.show()
    # data = figx.axes[0].lines[0].get_data()

# --- VISUALIZING SCENE AND RAYS ---
def visualize_scene_rays(scene,rays,image_path,axis_limits = [[-1,1],[-1,1],[-1,1]]):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot all shapes of the scene
    plot_scene_in_ax(scene,ax)

    # plot the rays
    ray_color = 'black'
    for index_ray in np.arange(start=0,stop=len(rays)):
        ax.scatter3D(rays[index_ray].o.x,rays[index_ray].o.y,rays[index_ray].o.z,c=ray_color)
        ax.plot([rays[index_ray].o.x, rays[index_ray].o.x+rays[index_ray].d.x*0.01],[rays[index_ray].o.y, rays[index_ray].o.y+rays[index_ray].d.y*0.01],[rays[index_ray].o.z, rays[index_ray].o.z+rays[index_ray].d.z*0.01],color=ray_color)     
    
    # add axis labels & save the figure
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_zlim(axis_limits[2])
    ax.set_box_aspect(aspect=(1,(axis_limits[1][1]-axis_limits[1][0])/(axis_limits[0][1]-axis_limits[0][0]),(axis_limits[2][1]-axis_limits[2][0])/(axis_limits[0][1]-axis_limits[0][0])))
    image_file_name = image_path / 'scene_ray_visualization.jpeg'
    plt.savefig(image_file_name.resolve(),dpi=600)
    pickle.dump(fig, open((image_path / 'scene_ray_visualization.fig.pickle').resolve(), 'wb'))
    
    # Loading of pickle result
    # import pickle
    # import matplotlib.pyplot as plt
    # figx = pickle.load(open('E:/Max/02-projects/202105-Ultrasound_mb/02-simulations_ray_tracing/20210909-US_modeling_ray_tracing_transmission_membrane/scene_ray_visualization.fig.pickle','rb'))
    # figx.show()
    # data = figx.axes[0].lines[0].get_data()


# --- PLOTTING ALL SHAPES OF A SCENE ---
def plot_scene_in_ax(scene,ax):
    for shape in scene.shapes():
        type_shape = shape.type()
        face_color = 'lightsalmon'
        if shape.is_emitter():
            face_color = 'deepskyblue'
        elif shape.is_sensor():
            face_color = 'greenyellow'

        if type_shape == "cylinder":
            # Set of all spherical angles:
            num_surface_points = 100
            u = np.linspace(0, 2 * np.pi, num_surface_points)
            
            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            x_ground = np.outer(np.cos(u), np.ones_like(u))
            y_ground = np.outer(np.sin(u), np.cos(u))
            z_ground = np.zeros(x.shape)
            x_side = np.outer(np.cos(u), np.ones_like(u))
            y_side = np.outer(np.sin(u), np.ones_like(u))
            z_side = np.outer(np.ones_like(u), np.linspace(0,1,num_surface_points))
            x_top = np.outer(np.cos(u), np.ones_like(u))
            y_top = np.outer(np.sin(u), np.cos(u))
            z_top = np.ones(x.shape)

            for index_row in range(0,x.shape[0]):
                for index_col in range(0,x.shape[1]):
                    point_transformed_ground = shape.to_world().transform_point(Point3f(x_ground[index_row,index_col],y_ground[index_row,index_col],z_ground[index_row,index_col]))
                    x_ground[index_row,index_col] = point_transformed_ground.x
                    y_ground[index_row,index_col] = point_transformed_ground.y
                    z_ground[index_row,index_col] = point_transformed_ground.z
                    point_transformed_side = shape.to_world().transform_point(Point3f(x_side[index_row,index_col],y_side[index_row,index_col],z_side[index_row,index_col]))
                    x_side[index_row,index_col] = point_transformed_side.x
                    y_side[index_row,index_col] = point_transformed_side.y
                    z_side[index_row,index_col] = point_transformed_side.z
                    point_transformed_top = shape.to_world().transform_point(Point3f(x_top[index_row,index_col],y_top[index_row,index_col],z_top[index_row,index_col]))
                    x_top[index_row,index_col] = point_transformed_top.x
                    y_top[index_row,index_col] = point_transformed_top.y
                    z_top[index_row,index_col] = point_transformed_top.z
            
            # Plot surface
            ax.plot_surface(x_ground, y_ground, z_ground,  rstride=4, cstride=4, color=face_color,alpha=.5)
            ax.plot_surface(x_side, y_side, z_side,  rstride=4, cstride=4, color=face_color,alpha=.5)
            ax.plot_surface(x_top, y_top, z_top,  rstride=4, cstride=4, color=face_color,alpha=.5)
        elif type_shape == "rectangle":
            # apply homogeneous transformation on edges of rectangle
            edge_lower_left = shape.to_world().transform_point(Point3f(-1.,-1.,0.))
            edge_lower_right = shape.to_world().transform_point(Point3f(1.,-1.,0.))
            edge_upper_right = shape.to_world().transform_point(Point3f(1.,1.,0.))
            edge_upper_left = shape.to_world().transform_point(Point3f(-1.,1.,0.))

            # draw rectangle in 3D
            ax.add_collection3d(art3d.Poly3DCollection([[edge_lower_left,edge_lower_right,edge_upper_right,edge_upper_left]], closed=True,facecolors=face_color, linewidths=1, edgecolors='None', alpha=.75))
        elif type_shape == "disk":
            # Set of all spherical angles:
            num_surface_points = 100
            u = np.linspace(0, 2 * np.pi, num_surface_points)
            
            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            x = np.outer(np.cos(u), np.ones_like(u))
            y = np.outer(np.sin(u), np.cos(u))
            z = np.zeros(x.shape)

            for index_row in range(0,x.shape[0]):
                for index_col in range(0,x.shape[1]):
                    point_transformed = shape.to_world().transform_point(Point3f(x[index_row,index_col],y[index_row,index_col],z[index_row,index_col]))
                    x[index_row,index_col] = point_transformed.x
                    y[index_row,index_col] = point_transformed.y
                    z[index_row,index_col] = point_transformed.z
            
            # Plot surface
            ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=face_color,alpha=.5)
        elif type_shape == "sphere":
            # Set of all spherical angles:
            num_surface_points = 100
            u = np.linspace(0, 2 * np.pi, num_surface_points)
            v = np.linspace(0, np.pi, num_surface_points)

            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))

            for index_row in range(0,x.shape[0]):
                for index_col in range(0,x.shape[1]):
                    point_transformed = shape.to_world().transform_point(Point3f(x[index_row,index_col],y[index_row,index_col],z[index_row,index_col]))
                    x[index_row,index_col] = point_transformed.x
                    y[index_row,index_col] = point_transformed.y
                    z[index_row,index_col] = point_transformed.z
            
            # Plot surface
            ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=face_color,alpha=.5)
        else:
            print("Visualization for shape type '", type_shape, "' not implemented.") 
