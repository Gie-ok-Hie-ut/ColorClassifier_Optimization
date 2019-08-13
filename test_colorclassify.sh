#CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /data1/5K_small/processed/ --name colorenhance_resonly --model colorenhance --phase test --no_dropout --dataset_mode aligned_test $@
#CUDA_VISIBLE_DEVICES=5 python test.py --dataroot /data1/victorleee/5k_resized/ --name colorclassify_pretrained_nasnetalarge --model colorclassify --loadSize 340 --fineSize 331 --phase test --no_dropout --dataset_mode fivek $@
#CUDA_VISIBLE_DEVICES=5 python test.py --dataroot /data1/victorleee/5k_resized/ --name colorclassify_pretrained_nasnetalarge --model colorclassify --loadSize 340 --fineSize 331 --phase test --no_dropout --dataset_mode fivek $@
#CUDA_VISIBLE_DEVICES=5 python test.py --dataroot /data1/victorleee/5k_resized/ --name colorclassify_pretrained_vgg19_AB --model colorclassify --loadSize 256 --fineSize 224 --phase test --no_dropout --dataset_mode fivek2 $@
#CUDA_VISIBLE_DEVICES=5 python test.py --dataroot /data1/victorleee/5k_resized/ --name colorclassify_pretrained_vgg19_BE --model colorclassify --loadSize 256 --fineSize 224  --phase test --no_dropout --dataset_mode fivek2 $@
#CUDA_VISIBLE_DEVICES=6 python test.py --dataroot /data1/victorleee/5k_resized/ --name colorclassify_pretrained_vgg19_ABCDE --model colorclassify --loadSize 256 --fineSize 224  --phase test --no_dropout --dataset_mode fivek $@
CUDA_VISIBLE_DEVICES=3 python test.py --dataroot /data1/victorleee/5k_resized/ --name colorclassify_aadb_histogram --model colorclassify2 --loadSize 256 --fineSize 224  --no_dropout --dataset_mode aadb $@



