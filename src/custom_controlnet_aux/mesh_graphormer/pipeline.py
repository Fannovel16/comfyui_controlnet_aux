import os
import torch
import gc
import numpy as np
from custom_controlnet_aux.mesh_graphormer.depth_preprocessor import Preprocessor

import torchvision.models as models
from custom_mesh_graphormer.modeling.bert import BertConfig, Graphormer
from custom_mesh_graphormer.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from custom_mesh_graphormer.modeling._mano import MANO, Mesh
from custom_mesh_graphormer.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from custom_mesh_graphormer.modeling.hrnet.config import config as hrnet_config
from custom_mesh_graphormer.modeling.hrnet.config import update_config as hrnet_update_config
from custom_mesh_graphormer.utils.miscellaneous import set_seed
from argparse import Namespace
from pathlib import Path
import cv2
from torchvision import transforms
import numpy as np
import cv2
from trimesh import Trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms
from pathlib import Path
from custom_controlnet_aux.util import custom_hf_download
import custom_mesh_graphormer
from comfy.model_management import soft_empty_cache
from packaging import version

args = Namespace(
    num_workers=4,
    img_scale_factor=1,
    image_file_or_path=os.path.join('', 'MeshGraphormer', 'samples', 'hand'), 
    model_name_or_path=str(Path(custom_mesh_graphormer.__file__).parent / "modeling/bert/bert-base-uncased"),
    resume_checkpoint=None,
    output_dir='output/',
    config_name='',
    a='hrnet-w64',
    arch='hrnet-w64',
    num_hidden_layers=4,
    hidden_size=-1,
    num_attention_heads=4,
    intermediate_size=-1,
    input_feat_dim='2051,512,128',
    hidden_feat_dim='1024,256,64',
    which_gcn='0,0,1',
    mesh_type='hand',
    run_eval_only=True,
    device="cpu",
    seed=88,
    hrnet_checkpoint=custom_hf_download("hr16/ControlNet-HandRefiner-pruned", 'hrnetv2_w64_imagenet_pretrained.pth')
)

#Since mediapipe v0.10.5, the hand category has been correct
if version.parse(mp.__version__) >= version.parse('0.10.5'):
    true_hand_category = {"Right": "right", "Left": "left"}
else:
    true_hand_category = {"Right": "left", "Left": "right"}

class MeshGraphormerMediapipe(Preprocessor):
    def __init__(self, args=args, detect_thr=0.6, presence_thr=0.6) -> None:
        #global logger
        # Setup CUDA, GPU & distributed training
        args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
        print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

        #mkdir(args.output_dir)
        #logger = setup_logger("Graphormer", args.output_dir, get_rank())
        set_seed(args.seed, args.num_gpus)
        #logger.info("Using {} GPUs".format(args.num_gpus))

        # Mesh and MANO utils
        mano_model = MANO().to(args.device)
        mano_model.layer = mano_model.layer.to(args.device)
        mesh_sampler = Mesh(device=args.device)

        # Renderer for visualization
        # renderer = Renderer(faces=mano_model.face)

        # Load pretrained model
        trans_encoder = []

        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [3]

        # which encoder block to have graph convs
        which_blk_graph = [int(item) for item in args.which_gcn.split(',')]

        if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
            # if only run eval, load checkpoint
            #logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
            _model = torch.load(args.resume_checkpoint)

        else:
            # init three transformer-encoder blocks in a loop
            for i in range(len(output_feat_dim)):
                config_class, model_class = BertConfig, Graphormer
                config = config_class.from_pretrained(args.config_name if args.config_name \
                        else args.model_name_or_path, attn_implementation="eager")

                config.output_attentions = False
                config.img_feature_dim = input_feat_dim[i] 
                config.output_feature_dim = output_feat_dim[i]
                args.hidden_size = hidden_feat_dim[i]
                args.intermediate_size = int(args.hidden_size*2)

                if which_blk_graph[i]==1:
                    config.graph_conv = True
                    #logger.info("Add Graph Conv")
                else:
                    config.graph_conv = False

                config.mesh_type = args.mesh_type

                # update model structure if specified in arguments
                update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
                for idx, param in enumerate(update_params):
                    arg_param = getattr(args, param)
                    config_param = getattr(config, param)
                    if arg_param > 0 and arg_param != config_param:
                        #logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                        setattr(config, param, arg_param)

                # init a transformer encoder and append it to a list
                assert config.hidden_size % config.num_attention_heads == 0
                model = model_class(config=config) 
                #logger.info("Init model from scratch.")
                trans_encoder.append(model)
            
            # create backbone model
            if args.arch=='hrnet':
                hrnet_yaml = Path(__file__).parent / 'cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                hrnet_checkpoint = args.hrnet_checkpoint
                hrnet_update_config(hrnet_config, hrnet_yaml)
                backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
                #logger.info('=> loading hrnet-v2-w40 model')
            elif args.arch=='hrnet-w64':
                hrnet_yaml = Path(__file__).parent / 'cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                hrnet_checkpoint = args.hrnet_checkpoint
                hrnet_update_config(hrnet_config, hrnet_yaml)
                backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
                #logger.info('=> loading hrnet-v2-w64 model')
            else:
                print("=> using pre-trained model '{}'".format(args.arch))
                backbone = models.__dict__[args.arch](pretrained=True)
                # remove the last fc layer
                backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

            trans_encoder = torch.nn.Sequential(*trans_encoder)
            total_params = sum(p.numel() for p in trans_encoder.parameters())
            #logger.info('Graphormer encoders total parameters: {}'.format(total_params))
            backbone_total_params = sum(p.numel() for p in backbone.parameters())
            #logger.info('Backbone total parameters: {}'.format(backbone_total_params))

            # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
            _model = Graphormer_Network(args, config, backbone, trans_encoder)

            if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
                # for fine-tuning or resume training or inference, load weights from checkpoint
                #logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
                # workaround approach to load sparse tensor in graph conv.
                state_dict = torch.load(args.resume_checkpoint)
                _model.load_state_dict(state_dict, strict=False)
                del state_dict
                gc.collect()
                soft_empty_cache()

        # update configs to enable attention outputs
        setattr(_model.trans_encoder[-1].config,'output_attentions', True)
        setattr(_model.trans_encoder[-1].config,'output_hidden_states', True)
        _model.trans_encoder[-1].bert.encoder.output_attentions = True
        _model.trans_encoder[-1].bert.encoder.output_hidden_states =  True
        for iter_layer in range(4):
            _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
        for inter_block in range(3):
            setattr(_model.trans_encoder[-1].config,'device', args.device)

        _model.to(args.device)
        self._model = _model
        self.mano_model = mano_model
        self.mesh_sampler = mesh_sampler

        self.transform = transforms.Compose([           
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
        #Fix File loading is not yet supported on Windows
        with open(str( Path(__file__).parent / "hand_landmarker.task" ), 'rb') as file:
            model_data = file.read()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            min_hand_detection_confidence=detect_thr,
                                            min_hand_presence_confidence=presence_thr,
                                            min_tracking_confidence=0.6,
                                            num_hands=2)

        self.detector = vision.HandLandmarker.create_from_options(options)
        
    
    def get_rays(self, W, H, fx, fy, cx, cy, c2w_t, center_pixels): # rot = I
   
        j, i = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32))
        if center_pixels:
            i = i.copy() + 0.5
            j = j.copy() + 0.5

        directions = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], -1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

        rays_o = np.expand_dims(c2w_t,0).repeat(H*W, 0)

        rays_d = directions    # (H, W, 3)
        rays_d = (rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)).reshape(-1,3)

        return rays_o, rays_d
    
    def get_mask_bounding_box(self, extrema, H, W, padding=30, dynamic_resize=0.15):
        x_min, x_max, y_min, y_max = extrema
        bb_xpad = max(int((x_max - x_min + 1) * dynamic_resize), padding)
        bb_ypad = max(int((y_max - y_min + 1) * dynamic_resize), padding)
        bbx_min = np.max((x_min - bb_xpad, 0))
        bbx_max = np.min((x_max + bb_xpad, W-1))
        bby_min = np.max((y_min - bb_ypad, 0))
        bby_max = np.min((y_max + bb_ypad, H-1))
        return bbx_min, bbx_max, bby_min, bby_max

    def run_inference(self, img, Graphormer_model, mano, mesh_sampler, scale, crop_len):
        global args
        H, W = int(crop_len), int(crop_len)
        Graphormer_model.eval()
        mano.eval()
        device = next(Graphormer_model.parameters()).device
        with torch.no_grad():
            img_tensor = self.transform(img)
            batch_imgs = torch.unsqueeze(img_tensor, 0).to(device)
            
            # forward-pass
            pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = Graphormer_model(batch_imgs, mano, mesh_sampler)

            # obtain 3d joints, which are regressed from the full mesh
            pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
            # obtain 2d joints, which are projected from 3d joints of mesh
            #pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
            #pred_2d_coarse_vertices_from_mesh = orthographic_projection(pred_vertices_sub.contiguous(), pred_camera.contiguous())
            pred_camera = pred_camera.cpu()
            pred_vertices = pred_vertices.cpu()
            mesh = Trimesh(vertices=pred_vertices[0], faces=mano.face)
            res = crop_len
            focal_length = 1000 * scale
            camera_t = np.array([-pred_camera[1], -pred_camera[2], -2*focal_length/(res * pred_camera[0] +1e-9)])
            pred_3d_joints_camera = pred_3d_joints_from_mesh.cpu()[0] - camera_t
            z_3d_dist = pred_3d_joints_camera[:,2].clone()

            pred_2d_joints_img_space = ((pred_3d_joints_camera/z_3d_dist[:,None]) * np.array((focal_length, focal_length, 1)))[:,:2] + np.array((W/2, H/2))

            rays_o, rays_d = self.get_rays(W, H, focal_length, focal_length, W/2, H/2, camera_t, True)
            coords = np.array(list(np.ndindex(H,W))).reshape(H,W,-1).transpose(1,0,2).reshape(-1,2)
            intersector = RayMeshIntersector(mesh)
            points, index_ray, _ = intersector.intersects_location(rays_o, rays_d, multiple_hits=False)

            tri_index = intersector.intersects_first(rays_o, rays_d)

            tri_index = tri_index[index_ray]

            assert len(index_ray) == len(tri_index)
            
            discriminator = (np.sum(mesh.face_normals[tri_index]* rays_d[index_ray], axis=-1)<= 0)
            points = points[discriminator] # ray intesects in interior faces, discard them

            if len(points) == 0:
                return None, None
            depth = (points + camera_t)[:,-1]
            index_ray = index_ray[discriminator]
            pixel_ray = coords[index_ray]

            minval = np.min(depth)
            maxval = np.max(depth)
            depthmap = np.zeros([H,W])

            depthmap[pixel_ray[:, 0], pixel_ray[:, 1]] = 1.0 - (0.8 * (depth - minval) / (maxval - minval))
            depthmap *= 255
        return depthmap, pred_2d_joints_img_space


    def get_depth(self, np_image, padding):
        info = {}

        # STEP 3: Load the input image.
        #https://stackoverflow.com/a/76407270
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image.copy())

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = self.detector.detect(image)

        handedness_list = detection_result.handedness
        hand_landmarks_list = detection_result.hand_landmarks

        raw_image = image.numpy_view()
        H, W, C = raw_image.shape


        # HANDLANDMARKS CAN BE EMPTY, HANDLE THIS!
        if len(hand_landmarks_list) == 0:
            return None, None, None
        raw_image = raw_image[:, :, :3]

        padded_image = np.zeros((H*2, W*2, 3))
        padded_image[int(1/2 * H):int(3/2 * H), int(1/2 * W):int(3/2 * W)] = raw_image

        hand_landmarks_list, handedness_list = zip(
            *sorted(
                zip(hand_landmarks_list, handedness_list), key=lambda x: x[0][9].z, reverse=True
            )
        )

        padded_depthmap = np.zeros((H*2, W*2))
        mask = np.zeros((H, W))
        crop_boxes = []
        #bboxes = []
        groundtruth_2d_keypoints = []
        hands = []
        depth_failure = False
        crop_lens = []
        abs_boxes = []
        
        for idx in range(len(hand_landmarks_list)):
            hand = true_hand_category[handedness_list[idx][0].category_name]
            hands.append(hand)
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            height, width, _ = raw_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]

            # x_min, x_max, y_min, y_max: extrema from mediapipe keypoint detection
            x_min = int(min(x_coordinates) * width)
            x_max = int(max(x_coordinates) * width)
            x_c = (x_min + x_max)//2
            y_min = int(min(y_coordinates) * height)
            y_max = int(max(y_coordinates) * height)
            y_c = (y_min + y_max)//2
            abs_boxes.append([x_min, x_max, y_min, y_max])

            #if x_max - x_min < 60 or y_max - y_min < 60:
            #    continue

            crop_len = (max(x_max - x_min, y_max - y_min) * 1.6) //2 * 2

            # crop_x_min, crop_x_max, crop_y_min, crop_y_max: bounding box for mesh reconstruction 
            crop_x_min = int(x_c - (crop_len/2 - 1) + W/2)
            crop_x_max = int(x_c + crop_len/2 + W/2)
            crop_y_min = int(y_c - (crop_len/2 - 1) + H/2)
            crop_y_max = int(y_c + crop_len/2 + H/2)

            cropped = padded_image[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
            crop_boxes.append([crop_y_min, crop_y_max, crop_x_min, crop_x_max])
            crop_lens.append(crop_len)
            if hand == "left":
                cropped = cv2.flip(cropped, 1)

            if crop_len < 224:
                graphormer_input = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_CUBIC)
            else:
                graphormer_input = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
            scale = crop_len/224
            cropped_depthmap, pred_2d_keypoints = self.run_inference(graphormer_input.astype(np.uint8), self._model, self.mano_model, self.mesh_sampler, scale, int(crop_len)) 

            if cropped_depthmap is None:
                depth_failure = True
                break
            #keypoints_image_space = pred_2d_keypoints * (crop_y_max - crop_y_min + 1)/224
            groundtruth_2d_keypoints.append(pred_2d_keypoints)
            
            if hand == "left":
                cropped_depthmap = cv2.flip(cropped_depthmap, 1)
            resized_cropped_depthmap = cv2.resize(cropped_depthmap, (int(crop_len), int(crop_len)), interpolation=cv2.INTER_LINEAR)
            nonzero_y, nonzero_x = (resized_cropped_depthmap != 0).nonzero()
            if len(nonzero_y) == 0 or len(nonzero_x) == 0:
                depth_failure = True
                break
            padded_depthmap[crop_y_min+nonzero_y, crop_x_min+nonzero_x] = resized_cropped_depthmap[nonzero_y, nonzero_x]

            # nonzero stands for nonzero value on the depth map
            # coordinates of nonzero depth pixels in original image space
            original_nonzero_x = crop_x_min+nonzero_x - int(W/2)
            original_nonzero_y = crop_y_min+nonzero_y - int(H/2)
            
            nonzerox_min = min(np.min(original_nonzero_x), x_min)
            nonzerox_max = max(np.max(original_nonzero_x), x_max)
            nonzeroy_min = min(np.min(original_nonzero_y), y_min)
            nonzeroy_max = max(np.max(original_nonzero_y), y_max)

            bbx_min, bbx_max, bby_min, bby_max = self.get_mask_bounding_box((nonzerox_min, nonzerox_max, nonzeroy_min, nonzeroy_max), H, W, padding)
            mask[bby_min:bby_max+1, bbx_min:bbx_max+1] = 1.0
            #bboxes.append([int(bbx_min), int(bbx_max), int(bby_min), int(bby_max)])
        if depth_failure:
            #print("cannot detect normal hands")
            return None, None, None
        depthmap = padded_depthmap[int(1/2 * H):int(3/2 * H), int(1/2 * W):int(3/2 * W)].astype(np.uint8)
        mask = (255.0 * mask).astype(np.uint8)
        info["groundtruth_2d_keypoints"] = groundtruth_2d_keypoints
        info["hands"] = hands
        info["crop_boxes"] = crop_boxes
        info["crop_lens"] = crop_lens
        info["abs_boxes"] = abs_boxes
        return depthmap, mask, info
    
    def get_keypoints(self, img, Graphormer_model, mano, mesh_sampler, scale, crop_len):
        global args
        H, W = int(crop_len), int(crop_len)
        Graphormer_model.eval()
        mano.eval()
        device = next(Graphormer_model.parameters()).device
        with torch.no_grad():
            img_tensor = self.transform(img)
            #print(img_tensor)
            batch_imgs = torch.unsqueeze(img_tensor, 0).to(device)
            
            # forward-pass
            pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = Graphormer_model(batch_imgs, mano, mesh_sampler)

            # obtain 3d joints, which are regressed from the full mesh
            pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
            # obtain 2d joints, which are projected from 3d joints of mesh
            #pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
            #pred_2d_coarse_vertices_from_mesh = orthographic_projection(pred_vertices_sub.contiguous(), pred_camera.contiguous())
            pred_camera = pred_camera.cpu()
            pred_vertices = pred_vertices.cpu()
            #
            res = crop_len
            focal_length = 1000 * scale
            camera_t = np.array([-pred_camera[1], -pred_camera[2], -2*focal_length/(res * pred_camera[0] +1e-9)])
            pred_3d_joints_camera = pred_3d_joints_from_mesh.cpu()[0] - camera_t
            z_3d_dist = pred_3d_joints_camera[:,2].clone()
            pred_2d_joints_img_space = ((pred_3d_joints_camera/z_3d_dist[:,None]) * np.array((focal_length, focal_length, 1)))[:,:2] + np.array((W/2, H/2))
            
        return pred_2d_joints_img_space
    

    def eval_mpjpe(self, sample, info):
        H, W, C = sample.shape
        padded_image = np.zeros((H*2, W*2, 3))
        padded_image[int(1/2 * H):int(3/2 * H), int(1/2 * W):int(3/2 * W)] = sample
        crop_boxes = info["crop_boxes"]
        hands = info["hands"]
        groundtruth_2d_keypoints = info["groundtruth_2d_keypoints"]
        crop_lens = info["crop_lens"]
        pjpe = 0
        for i in range(len(crop_boxes)):#box in crop_boxes:
            crop_y_min, crop_y_max, crop_x_min, crop_x_max = crop_boxes[i]
            cropped = padded_image[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
            hand = hands[i]
            if hand == "left":
                cropped = cv2.flip(cropped, 1)
            crop_len = crop_lens[i]
            scale = crop_len/224
            if crop_len < 224:
                graphormer_input = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_CUBIC)
            else:
                graphormer_input = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
            generated_keypoint = self.get_keypoints(graphormer_input.astype(np.uint8), self._model, self.mano_model, self.mesh_sampler, scale, crop_len)
            #generated_keypoint = generated_keypoint * ((crop_y_max - crop_y_min + 1)/224)
            pjpe += np.sum(np.sqrt(np.sum(((generated_keypoint - groundtruth_2d_keypoints[i]) ** 2).numpy(), axis=1)))
            pass
        mpjpe = pjpe/(len(crop_boxes) * 21)
        return mpjpe
