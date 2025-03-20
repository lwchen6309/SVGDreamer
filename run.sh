prompt="A poster of the great wall, teal and orange color scheme, autumn colors"
python svgdreamer.py x=iconography "prompt='$prompt'" result_path='./logs/great_wall' state.mprec='fp16' \
    x.vpsd.n_particle=1 x.vpsd.vsd_n_particle=1 x.vpsd.phi_n_particle=1 
 
# python svgdreamer.py x=iconography "prompt='Sydney opera house. oil painting. by Van Gogh'" result_path='./logs/SydneyOperaHouse-OilPainting' state.mprec='fp16'

# python svgdreamer.py x=iconography skip_sive=False "prompt='$prompt'" \
#     token_ind=4 x.vpsd.t_schedule='randint' result_path='./logs/great_wall' \
#     x.xing_loss.weight=0. x.num_paths=2048 +target_file="./init_target/demo_2048/example_0072_demo.svg" \
#     x.composition_loss.weight=0. x.sam_composition_loss.weight=0. x.vpsd.num_iter=2000

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
