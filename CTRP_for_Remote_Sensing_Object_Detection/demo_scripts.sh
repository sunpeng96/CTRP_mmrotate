demo(){
  python demo/image_demo.py demo/input_image.png \
    demo/demo_for_remote_sensing_object_detection.py \
    demo/checkpoint_for_remote_sensing_object_detection.pth \
    --out-file demo/detection_result.png
}

demo
