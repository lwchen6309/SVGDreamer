prompt="A poster of the great wall, teal and orange color scheme, autumn colors, ultra realistic"
# composition_type=("golden_spiral" "pyramid" "diagonal" "l_shape")

python svgdreamer.py x=iconography skip_sive=False "prompt='$prompt'" \
    token_ind=4 x.vpsd.t_schedule='randint' result_path='./logs/real_great_wall' \
    x.xing_loss.weight=0. x.num_paths=2048 +target_file="./init_target/demo_2048/example_0072_demo.svg" \
    x.composition_loss.weight=0. x.sam_composition_loss.weight=0.


# composition_type=("golden_spiral" "pyramid" "diagonal" "l_shape")
# weight=5e2
# sam_weight=0
# sigma=75

# for composition in "${composition_type[@]}"; do
#     python svgdreamer.py x=iconography skip_sive=False "prompt='$prompt'" \
#     token_ind=4 x.vpsd.t_schedule='randint' result_path='./logs/real_great_wall' \
#     x.xing_loss.weight=0. x.num_paths=512 +target_file="./init_target/demo/example_0072_demo.svg" \
#     x.composition_loss.composition_type=$composition x.composition_loss.weight=$weight x.composition_loss.sigma=$sigma \
#     x.sam_composition_loss.weight=$sam_weight
# done
