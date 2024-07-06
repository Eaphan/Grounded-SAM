
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

nuscenes_dir=../datasets/nuscenes
nuscenes_output=nuscenes_seg_grounded_sam_output

if [ -e sam_hq_vit_h.pth ]; then
    echo "sam_hq_vit_h.pth exists."
else
    echo "Please download sam_hq_vit_h.pth: https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?pli=1"
    echo "Please download sam_hq_vit_h.pth: https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?pli=1"
    echo "Please download sam_hq_vit_h.pth: https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?pli=1"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python infer.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint ./sam_hq_vit_h.pth \
  --use_sam_hq \
  --input_dir ${nuScenes_dir}/samples/CAM_BACK \
  --output_dir ${nuscenes_output}/samples/CAM_BACK \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --text_prompt "barrier.bicycle.bus.car.motorcycle.pedestrian.traffic cone.truck.road.sidewalk.terrain.vegetation.building." \
  --device "cuda"

python infer.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint ./sam_hq_vit_h.pth \
  --use_sam_hq \
  --input_dir ${nuScenes_dir}/samples/CAM_BACK_LEFT \
  --output_dir ${nuscenes_output}/samples/CAM_BACK_LEFT \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --text_prompt "barrier.bicycle.bus.car.motorcycle.pedestrian.traffic cone.truck.road.sidewalk.terrain.vegetation.building." \
  --device "cuda"

python infer.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint ./sam_hq_vit_h.pth \
  --use_sam_hq \
  --input_dir ${nuScenes_dir}/samples/CAM_BACK_RIGHT \
  --output_dir ${nuscenes_output}/samples/CAM_BACK_RIGHT \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --text_prompt "barrier.bicycle.bus.car.motorcycle.pedestrian.traffic cone.truck.road.sidewalk.terrain.vegetation.building." \
  --device "cuda"

python infer.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint ./sam_hq_vit_h.pth \
  --use_sam_hq \
  --input_dir ${nuScenes_dir}/samples/CAM_FRONT \
  --output_dir ${nuscenes_output}/samples/CAM_FRONT \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --text_prompt "barrier.bicycle.bus.car.motorcycle.pedestrian.traffic cone.truck.road.sidewalk.terrain.vegetation.building." \
  --device "cuda"

python infer.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint ./sam_hq_vit_h.pth \
  --use_sam_hq \
  --input_dir ${nuScenes_dir}/samples/CAM_FRONT_LEFT \
  --output_dir ${nuscenes_output}/samples/CAM_FRONT_LEFT \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --text_prompt "barrier.bicycle.bus.car.motorcycle.pedestrian.traffic cone.truck.road.sidewalk.terrain.vegetation.building." \
  --device "cuda"

python infer.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint ./sam_hq_vit_h.pth \
  --use_sam_hq \
  --input_dir ${nuScenes_dir}/samples/CAM_FRONT_RIGHT \
  --output_dir ${nuscenes_output}/samples/CAM_FRONT_RIGHT \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --text_prompt "barrier.bicycle.bus.car.motorcycle.pedestrian.traffic cone.truck.road.sidewalk.terrain.vegetation.building." \
  --device "cuda"

python change_format.py
