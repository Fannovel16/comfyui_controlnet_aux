"""
Rendering tools for 3D mesh visualization on 2D image.

Parts of the code are taken from https://github.com/akanazawa/hmr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import code
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight
import random


# Rotate the points by a specified angle.
def rotateY(points, angle):
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)

def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': (197, 27, 125),  # L lower leg
        'light_pink': (233, 163, 201),  # L upper leg
        'light_green': (161, 215, 106),  # L lower arm
        'green': (77, 146, 33),  # L upper arm
        'red': (215, 48, 39),  # head
        'light_red': (252, 146, 114),  # head
        'light_orange': (252, 141, 89),  # chest
        'purple': (118, 42, 131),  # R lower leg
        'light_purple': (175, 141, 195),  # R upper
        'light_blue': (145, 191, 219),  # R lower arm
        'blue': (69, 117, 180),  # R upper arm
        'gray': (130, 130, 130),  #
        'white': (255, 255, 255),  #
    }

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 14:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    elif joints.shape[1] == 21:  # hand
        parents = np.array([
            -1,
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            0,
            13,
            14,
            15,
            0,
            17,
            18,
            19,
        ])
        ecolors = {
            0: 'light_purple',
            1: 'light_green',
            2: 'light_green',
            3: 'light_green',
            4: 'light_green',
            5: 'pink',
            6: 'pink',
            7: 'pink',
            8: 'pink',
            9: 'light_blue',
            10: 'light_blue',
            11: 'light_blue',
            12: 'light_blue',
            13: 'light_red',
            14: 'light_red',
            15: 'light_red',
            16: 'light_red',
            17: 'purple',
            18: 'purple',
            19: 'purple',
            20: 'purple',
        }
    else:
        print('Unknown skeleton!!')

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image

def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = (255, 255, 0)
    margin = 15
    start_x = 5
    start_y = margin
    for key in sorted(content.keys()):
        text = "%s: %.2g" % (key, content[key])
        cv2.putText(image, text, (start_x, start_y), 0, 0.45, black)
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image

def visualize_reconstruction(img, img_size, gt_kp, vertices, pred_kp, camera, renderer, color='pink', focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    gt_vis = gt_kp[:, 2].astype(bool)
    loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
    debug_text = {"sc": camera[0], "tx": camera[1], "ty": camera[2], "kpl": loss}
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices, camera_t=camera_t,
                               img=img, use_bg=True,
                               focal_length=focal_length,
                               body_color=color)
    rend_img = draw_text(rend_img, debug_text)

    # Draw skeleton
    gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * img_size
    pred_joint = ((pred_kp + 1) * 0.5) * img_size
    img_with_gt = draw_skeleton( img, gt_joint, draw_edges=False, vis=gt_vis)
    skel_img = draw_skeleton(img_with_gt, pred_joint)

    combined = np.hstack([skel_img, rend_img])

    return combined

def visualize_reconstruction_test(img, img_size, gt_kp, vertices, pred_kp, camera, renderer, score, color='pink', focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    gt_vis = gt_kp[:, 2].astype(bool)
    loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
    debug_text = {"sc": camera[0], "tx": camera[1], "ty": camera[2], "kpl": loss, "pa-mpjpe": score*1000}
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices, camera_t=camera_t,
                               img=img, use_bg=True,
                               focal_length=focal_length,
                               body_color=color)
    rend_img = draw_text(rend_img, debug_text)

    # Draw skeleton
    gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * img_size
    pred_joint = ((pred_kp + 1) * 0.5) * img_size
    img_with_gt = draw_skeleton( img, gt_joint, draw_edges=False, vis=gt_vis)
    skel_img = draw_skeleton(img_with_gt, pred_joint)

    combined = np.hstack([skel_img, rend_img])

    return combined



def visualize_reconstruction_and_att(img, img_size, vertices_full, vertices, vertices_2d, camera, renderer, ref_points, attention, focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices_full, camera_t=camera_t,
                               img=img, use_bg=True, 
                               focal_length=focal_length, body_color='light_blue')


    heads_num, vertex_num, _ = attention.shape

    all_head = np.zeros((vertex_num,vertex_num))

    ###### find max
    # for i in range(vertex_num):
    #     for j in range(vertex_num):
    #         all_head[i,j] = np.max(attention[:,i,j])

    ##### find avg
    for h in range(4):
        att_per_img = attention[h]
        all_head = all_head + att_per_img   
    all_head = all_head/4

    col_sums = all_head.sum(axis=0)
    all_head = all_head / col_sums[np.newaxis, :]


    # code.interact(local=locals())

    combined = []
    if vertex_num>400:  # body
        selected_joints = [6,7,4,5,13] # [6,7,4,5,13,12] 
    else: # hand  
        selected_joints = [0, 4, 8, 12, 16, 20]
    # Draw attention
    for ii in range(len(selected_joints)):
        reference_id = selected_joints[ii]
        ref_point = ref_points[reference_id]
        attention_to_show = all_head[reference_id][14::] 
        min_v = np.min(attention_to_show)
        max_v = np.max(attention_to_show)
        norm_attention_to_show = (attention_to_show - min_v)/(max_v-min_v)

        vertices_norm = ((vertices_2d + 1) * 0.5) * img_size
        ref_norm = ((ref_point + 1) * 0.5) * img_size
        image = np.zeros_like(rend_img)

        for jj in range(vertices_norm.shape[0]):
            x = int(vertices_norm[jj,0])
            y = int(vertices_norm[jj,1])
            cv2.circle(image,(x,y), 1, (255,255,255), -1) 

        total_to_draw = []
        for jj in range(vertices_norm.shape[0]):
            thres = 0.0
            if norm_attention_to_show[jj]>thres:
                things = [norm_attention_to_show[jj], ref_norm, vertices_norm[jj]]
                total_to_draw.append(things)
                # plot_one_line(ref_norm, vertices_norm[jj], image, reference_id, alpha=0.4*(norm_attention_to_show[jj]-thres)/(1-thres)  )
        total_to_draw.sort()
        max_att_score = total_to_draw[-1][0]
        for item in total_to_draw:
            attention_score = item[0]
            ref_point = item[1]
            vertex = item[2]
            plot_one_line(ref_point, vertex, image, ii, alpha=(attention_score-thres)/(max_att_score-thres)  )
        # code.interact(local=locals())
        if len(combined)==0:
            combined = image
        else:
            combined = np.hstack([combined, image])

    final = np.hstack([img, combined, rend_img])

    return final


def visualize_reconstruction_and_att_local(img, img_size, vertices_full, vertices, vertices_2d, camera, renderer, ref_points, attention, color='light_blue', focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices_full, camera_t=camera_t,
                               img=img, use_bg=True, 
                               focal_length=focal_length, body_color=color)
    heads_num, vertex_num, _ = attention.shape
    all_head = np.zeros((vertex_num,vertex_num))

    ##### compute avg attention for 4 attention heads
    for h in range(4):
        att_per_img = attention[h]
        all_head = all_head + att_per_img   
    all_head = all_head/4

    col_sums = all_head.sum(axis=0)
    all_head = all_head / col_sums[np.newaxis, :]

    combined = []
    if vertex_num>400:  # body
        selected_joints = [7]  # [6,7,4,5,13,12] 
    else: # hand  
        selected_joints = [0] # [0, 4, 8, 12, 16, 20] 
    # Draw attention
    for ii in range(len(selected_joints)):
        reference_id = selected_joints[ii]
        ref_point = ref_points[reference_id]
        attention_to_show = all_head[reference_id][14::] 
        min_v = np.min(attention_to_show)
        max_v = np.max(attention_to_show)
        norm_attention_to_show = (attention_to_show - min_v)/(max_v-min_v)
        vertices_norm = ((vertices_2d + 1) * 0.5) * img_size
        ref_norm = ((ref_point + 1) * 0.5) * img_size
        image = rend_img*0.4

        total_to_draw = []
        for jj in range(vertices_norm.shape[0]):
            thres = 0.0
            if norm_attention_to_show[jj]>thres:
                things = [norm_attention_to_show[jj], ref_norm, vertices_norm[jj]]
                total_to_draw.append(things)
        total_to_draw.sort()
        max_att_score = total_to_draw[-1][0]
        for item in total_to_draw:
            attention_score = item[0]
            ref_point = item[1]
            vertex = item[2]
            plot_one_line(ref_point, vertex, image, ii, alpha=(attention_score-thres)/(max_att_score-thres)  )

        for jj in range(vertices_norm.shape[0]):
            x = int(vertices_norm[jj,0])
            y = int(vertices_norm[jj,1])
            cv2.circle(image,(x,y), 1, (255,255,255), -1) 

        if len(combined)==0:
            combined = image
        else:
            combined = np.hstack([combined, image])

    final = np.hstack([img, combined, rend_img])

    return final


def visualize_reconstruction_no_text(img, img_size, vertices, camera, renderer, color='pink', focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices, camera_t=camera_t,
                               img=img, use_bg=True,
                               focal_length=focal_length,
                               body_color=color)


    combined = np.hstack([img, rend_img])

    return combined


def plot_one_line(ref, vertex, img, color_index, alpha=0.0, line_thickness=None):
    # 13,6,7,8,3,4,5
    # att_colors = [(255, 221, 104), (255, 255, 0), (255, 215, 227),  (210, 240, 119), \
    #          (209, 238, 245), (244, 200, 243),  (233, 242, 216)] 
    att_colors = [(255, 255, 0), (244, 200, 243),  (210, 243, 119), (209, 238, 255), (200, 208, 255), (250, 238, 215)] 


    overlay = img.copy()
    # output = img.copy()
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    color = list(att_colors[color_index])
    c1, c2 = (int(ref[0]), int(ref[1])), (int(vertex[0]), int(vertex[1]))
    cv2.line(overlay, c1, c2, (alpha*float(color[0])/255,alpha*float(color[1])/255,alpha*float(color[2])/255) , thickness=tl, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)



def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


class Renderer(object):
    """
    Render mesh using OpenDR for visualization.
    """

    def __init__(self, width=800, height=600, near=0.5, far=1000, faces=None):
        self.colors = {'hand': [.9, .9, .9], 'pink': [.9, .7, .7], 'light_blue': [0.65098039, 0.74117647, 0.85882353] }
        self.width = width
        self.height = height
        self.faces = faces
        self.renderer = ColoredRenderer()

    def render(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if body_color is None:
            color = self.colors['light_blue']
        else:
            color = self.colors[body_color]

        if isinstance(self.renderer, TexturedRenderer):
            color = [1.,1.,1.]

        self.renderer.set(v=vertices, f=faces,
                          vc=color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r


    def render_vertex_color(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               vertex_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if vertex_color is None:
            vertex_color = self.colors['light_blue']


        self.renderer.set(v=vertices, f=faces,
                          vc=vertex_color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r