#!/bin/bash
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#partition name
#SBATCH --partition=viscam,svl
#################
#number of GPUs
#SBATCH --gres=gpu:1
##SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#################
#set a job name
#SBATCH --job-name="filmreconcontrastive"
#################
#a file for job output, you can check job progress, append the job ID with %j to make it unique
##SBATCH --output=test_output/%j.out
#################
# a file for errors from the job
##SBATCH --error=test_output/%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=336:00:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
# uncomment ##SBATCH --qos=long if you want your job to run longer than 48 hours, which is the default for normal partition,
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos, also change to normal partition
# since dev max run time is 2 hours.
##SBATCH --qos=long
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM)
##SBATCH -p dev
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=32G
##SBATCH --exclude=viscam1,svl11,svl10
##SBATCH --nodelist=viscam3
# CHANGE BATCH SIZE WHEN CHANGING GPUS
##SBATCH --gres=gpu:titanrtx:1
##SBATCH --gres=gpu:a40:1
#SBATCH --gres=gpu:2080ti:1
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 1
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
#SBATCH --mail-user=joycj@stanford.edu
# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
#now run normal batch commands
source /afs/cs.stanford.edu/u/joycj/.bashrc
conda activate /pasteur/u/joycj/conda/envs/film



train_questions="/viscam/u/joycj/dvr/final_datasets/cogen_multi_object_biased_005/questions/train/what_attr_is_the_attr_questions.h5"
val_questions="/viscam/u/joycj/dvr/final_datasets/cogen_multi_object/questions/val/what_attr_is_the_attr_questions.h5"
test_questions="/viscam/u/joycj/dvr/final_datasets/cogen_multi_object/questions/test/what_attr_is_the_attr_questions.h5"

train_features="/viscam/u/joycj/dvr/final_datasets/cogen_multi_object_biased_005/train/features.h5"
val_features="/viscam/u/joycj/dvr/final_datasets/cogen_multi_object/val/features.h5"
test_features="/viscam/u/joycj/dvr/final_datasets/cogen_multi_object/test/features.h5"

vocab="/viscam/u/joycj/dvr/final_datasets/cogen_multi_object/questions/train/what_attr_is_the_attr_vocab.json"

real_dataset_dir="/viscam/u/joycj/dvr/final_datasets/cogen_multi_object/real_test_images/"




checkpoint_path="/viscam/u/joycj/dvr/film/chckpt_paths/appendix_biased_005_film_moma_attr_ans_cogen_biased_005_78_uncond_stylegan_moma_attr_ans_cogen_test_81_gan_sampler_b8_t008.pt"

film_checkpoint_path="/viscam/u/joycj/dvr/film/checkpoint_paths/stage_1_film_moma_attr_ans_cogen_biased_005_78.pt"
stylegan_checkpoint_path="/viscam/u/joycj/dvr/film/checkpoint_paths/stage_2_uncond_stylegan_moma_attr_ans_cogen_test_81.pt"

# stage 1
# --training_film \

# stage 2
# --training_gan \

# stage 3
# --entropy_threshold 0.001 \
# --execution_engine_start_from $film_checkpoint_path \
# --program_generator_start_from $film_checkpoint_path \
# --stylegan_start_from $stylegan_checkpoint_path \


cd /viscam/u/joycj/dvr/fin_code/film
python fin_scripts/train_gan_sampler.py \
  --entropy_threshold 0.008 \
  --execution_engine_start_from $film_checkpoint_path \
  --program_generator_start_from $film_checkpoint_path \
  --stylegan_start_from $stylegan_checkpoint_path \
  --checkpoint_path $checkpoint_path \
  --model_type FiLMReconContrastiveIntermediate \
  --contrastive_loss TaskVGenContrastive \
  --contrastive_beta 0.1 \
  --r1_regularization_gamma 15.0 \
  --train_question_h5 $train_questions \
  --train_features_h5 $train_features \
  --val_question_h5 $test_questions \
  --val_features_h5 $test_features \
  --vocab_json $vocab \
  --real_dataset_dir $real_dataset_dir \
  --num_iterations 20000000 \
  --print_verbose_every 20000000 \
  --checkpoint_every 1000 \
  --record_loss_every 100 \
  --num_val_samples 149991 \
  --optimizer Adam \
  --learning_rate 1e-4 \
  --batch_size 8 \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 1 \
  --module_batchnorm 1 \
  --classifier_batchnorm 1 \
  --bidirectional 0 \
  --decoder_type linear \
  --encoder_type gru \
  --weight_decay 1e-5 \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 200 \
  --rnn_hidden_dim 4096 \
  --rnn_output_batchnorm 0 \
  --classifier_downsample maxpoolfull \
  --classifier_proj_dim 512 \
  --classifier_fc_dims 1024 \
  --module_input_proj 1 \
  --module_residual 1 \
  --module_dim 128 \
  --module_dropout 0e-2 \
  --module_stem_kernel_size 3 \
  --module_kernel_size 3 \
  --module_batchnorm_affine 0 \
  --module_num_layers 1 \
  --num_modules 4 \
  --condition_pattern 1,1,1,1 \
  --gamma_option linear \
  --gamma_baseline 1 \
  --use_gamma 1 \
  --use_beta 1 \
  --condition_method bn-film \
  --program_generator_parameter_efficient 1


echo "Done"
exit 0



