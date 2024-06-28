#!/bin/bash -xe

if [ -z "${MODEL_FILE}" ]; then
	MODEL_FILE=sd_xl_base_1.0.safetensors
fi
MODEL_FILE="/sd-models/${MODEL_FILE}"

if [ -z "${SAKURA_ARTIFACT_DIR}" ]; then
	SAKURA_ARTIFACT_DIR=/tmp
fi

function gen_image () {
	PROMPT="$1"
	NUM_IMAGES=1
	if [ -n "$2" ]; then
		NUM_IMAGES="$2"
	fi
	if [ -z "${SAMPLER}" ]; then
		SAMPLER="dpmsolver++"
	fi
	if [ -z "${STEPS}" ]; then
		STEPS=50
	fi
	if [ -z "${BATCH_SIZE}" ]; then
		BATCH_SIZE=4
	fi

	EXTRA_ARGS=""
	if [ -n "${LORA_URL}" ]; then
		LORA_FILENAME=`basename "${LORA_URL}"`
		curl -o "/opt/${LORA_FILENAME}" "${LORA_URL}"
		EXTRA_ARGS="--network_module networks.lora --network_weights /opt/${LORA_FILENAME}"
	fi
	if [ -n "${SEED}" ]; then
		EXTRA_ARGS=" --seed ${SEED}"
	fi

	python3 sdxl_gen_img.py \
		--ckpt ${MODEL_FILE} \
		--vae /sd-models/sdxl_vae.safetensors \
		--images_per_prompt ${NUM_IMAGES} ${EXTRA_ARGS} \
		--sampler ${SAMPLER} --steps ${STEPS} --batch_size ${BATCH_SIZE} \
		--outdir ${SAKURA_ARTIFACT_DIR} --xformers --fp16 --prompt "${PROMPT}"
}

function gen_lora () {
	DATA_URL="$1"
	OUTPUT_NAME="$2"
	CLASS_TOKENS="$3"

	if [ -z "${NUM_PROCESS}" ]; then
		NUM_PROCESS=2
	fi

	mkdir -p /opt/data
	curl ${DATA_URL} | tar zxvf - -C /opt/data

	eval "echo \"$(cat /opt/lora_config.toml.tmpl)\"" > /opt/lora_config.toml	

	if [ -z "${MAX_TRAIN_EPOCHS}" ]; then
		MAX_TRAIN_EPOCHS=10
	fi
	if [ -z "${BATCH_SIZE}" ]; then
		BATCH_SIZE=4
	fi

	accelerate launch --num_cpu_threads_per_process ${NUM_PROCESS} sdxl_train_network.py \
		--pretrained_model_name_or_path="${MODEL_FILE}" \
		--dataset_config='/opt/lora_config.toml' \
		--output_dir="${SAKURA_ARTIFACT_DIR}" \
		--output_name="${OUTPUT_NAME}" \
		--save_model_as=safetensors --prior_loss_weight=1.0 --resolution=1024,1024 \
		--train_batch_size=1 --train_batch_size=${BATCH_SIZE} --max_train_epochs=${MAX_TRAIN_EPOCHS} \
		--learning_rate=1e-4 \
		--xformers --mixed_precision="fp16" --cache_latents \
		--gradient_checkpointing --network_module=networks.lora --no_half_vae
}

nvidia-smi

TASK=$1

case "${TASK}" in
	"generate" ) gen_image "$2" "$3" ;;
	"lerning" ) gen_lora "$2" "$3" "$4" ;;
	"study" ) gen_lora "$2" "$3" "$4" ;;
esac
