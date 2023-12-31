from __future__ import absolute_import, division, print_function

import argparse
import os
import os.path as op
import code
import json
import zipfile
import torch
import numpy as np
from custom_mesh_graphormer.utils.metric_pampjpe import get_alignMesh


def load_pred_json(filepath):
    archive = zipfile.ZipFile(filepath, 'r')
    jsondata = archive.read('pred.json')
    reference = json.loads(jsondata.decode("utf-8"))
    return reference[0], reference[1]


def multiscale_fusion(output_dir):
    s = '10'
    filepath = output_dir+'ckpt200-sc10_rot0-pred.zip'
    ref_joints, ref_vertices = load_pred_json(filepath)
    ref_joints_array = np.asarray(ref_joints)
    ref_vertices_array = np.asarray(ref_vertices)

    rotations = [0.0]
    for i in range(1,10):
        rotations.append(i*10)
        rotations.append(i*-10)
    
    scale = [0.7,0.8,0.9,1.0,1.1]
    multiscale_joints = []
    multiscale_vertices = []

    counter = 0
    for s in scale:
        for r in rotations:
            setting = 'sc%02d_rot%s'%(int(s*10),str(int(r)))
            filepath = output_dir+'ckpt200-'+setting+'-pred.zip'
            joints, vertices = load_pred_json(filepath)
            joints_array = np.asarray(joints)
            vertices_array = np.asarray(vertices)

            pa_joint_error, pa_joint_array, _ = get_alignMesh(joints_array, ref_joints_array, reduction=None)
            pa_vertices_error, pa_vertices_array, _ = get_alignMesh(vertices_array, ref_vertices_array, reduction=None)
            print('--------------------------')
            print('scale:', s, 'rotate', r)
            print('PAMPJPE:', 1000*np.mean(pa_joint_error))
            print('PAMPVPE:', 1000*np.mean(pa_vertices_error))
            multiscale_joints.append(pa_joint_array)
            multiscale_vertices.append(pa_vertices_array)
            counter = counter + 1

    overall_joints_array = ref_joints_array.copy()
    overall_vertices_array = ref_vertices_array.copy()
    for i in range(counter):
        overall_joints_array += multiscale_joints[i]
        overall_vertices_array += multiscale_vertices[i]

    overall_joints_array /= (1+counter)
    overall_vertices_array /= (1+counter)
    pa_joint_error, pa_joint_array, _ = get_alignMesh(overall_joints_array, ref_joints_array, reduction=None)
    pa_vertices_error, pa_vertices_array, _ = get_alignMesh(overall_vertices_array, ref_vertices_array, reduction=None)
    print('--------------------------')
    print('overall:')
    print('PAMPJPE:', 1000*np.mean(pa_joint_error))
    print('PAMPVPE:', 1000*np.mean(pa_vertices_error))

    joint_output_save = overall_joints_array.tolist()
    mesh_output_save = overall_vertices_array.tolist()

    print('save results to pred.json')
    with open('pred.json', 'w') as f:
        json.dump([joint_output_save, mesh_output_save], f)


    filepath = output_dir+'ckpt200-multisc-pred.zip'
    resolved_submit_cmd = 'zip ' + filepath + '  ' +  'pred.json'
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    resolved_submit_cmd = 'rm pred.json'
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)


def run_multiscale_inference(model_path, mode, output_dir):
    
    if mode==True:
        rotations = [0.0]
        for i in range(1,10):
            rotations.append(i*10)
            rotations.append(i*-10)
        scale = [0.7,0.8,0.9,1.0,1.1]
    else:
        rotations = [0.0]
        scale = [1.0] 

    job_cmd = "python ./src/tools/run_gphmer_handmesh.py " \
            "--val_yaml freihand_v3/test.yaml " \
            "--resume_checkpoint %s " \
            "--per_gpu_eval_batch_size 32 --run_eval_only --num_worker 2 " \
            "--multiscale_inference " \
            "--rot %f " \
            "--sc %s " \
            "--arch hrnet-w64 " \
            "--num_hidden_layers 4 " \
            "--num_attention_heads 4 " \
            "--input_feat_dim 2051,512,128 " \
            "--hidden_feat_dim 1024,256,64 " \
            "--output_dir %s"

    for s in scale:
        for r in rotations:
            resolved_submit_cmd = job_cmd%(model_path, r, s, output_dir)
            print(resolved_submit_cmd)
            os.system(resolved_submit_cmd)

def main(args):
    model_path = args.model_path
    mode = args.multiscale_inference
    output_dir = args.output_dir
    run_multiscale_inference(model_path, mode, output_dir)
    if mode==True:
        multiscale_fusion(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint in the folder")
    parser.add_argument("--model_path")
    parser.add_argument("--multiscale_inference", default=False, action='store_true',) 
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    args = parser.parse_args()
    main(args)
