# python svgdreamer.py x=iconography skip_sive=False "prompt='A poster of the great wall, teal and orange color scheme, autumn colors'" token_ind=4 x.vpsd.t_schedule='randint' result_path='./logs/great_wall' x.xing_loss.weight=0. x.num_paths=512 +target_file="./init_target/demo/example_0072_demo.svg" x.composition_loss.weight=0. mv=True

weight=5e2
sam_weight=5e2
sigma=75
composition_type=("golden_spiral" "pyramid" "diagonal" "l_shape")
# composition_type=("golden_spiral")
for composition in "${composition_type[@]}"; do
    python svgdreamer.py x=iconography skip_sive=False "prompt='A poster of the great wall, teal and orange color scheme, autumn colors'" \
    token_ind=4 x.vpsd.t_schedule='randint' result_path='./logs/great_wall/sam_edge' \
    x.xing_loss.weight=0. x.num_paths=512 +target_file="./init_target/demo/example_0072_demo.svg" \
    x.composition_loss.composition_type=$composition x.composition_loss.weight=$weight x.composition_loss.sigma=$sigma \
    x.sam_composition_loss.weight=$sam_weight
done