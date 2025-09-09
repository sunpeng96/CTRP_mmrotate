demo(){
  python demo/image_demo.py demo/input_image.png \
    demo/demo_for_occluded_object_detection.py \
    demo/checkpoint_for_occluded_object_detection.pth \
    --out-file demo/detection_result.png --palette occluded_dota
    }

demo