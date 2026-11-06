SHELL := /bin/bash
.DEFAULT_GOAL := help

# Values come from .env when present (cp .env.example .env).
USER_NAME ?= $(shell grep -s '^USER_NAME=' .env | cut -d= -f2-)
ifeq ($(strip $(USER_NAME)),)
	USER_NAME := $(shell whoami)
endif

EXP ?= configs/experiments/cnn_clean.yaml
IMAGE := bamm-detectors-$(USER_NAME)
OS := $(shell uname -s)
BASE_DOCKERFILE ?= $(shell grep -s '^BASE_DOCKERFILE=' .env | cut -d= -f2-)
GPU_DEVICE ?= $(shell grep -s '^GPU_DEVICE=' .env | cut -d= -f2-)
DATASETS_DIR ?= $(shell grep -s '^DATASETS_DIR=' .env | cut -d= -f2-)

ifeq ($(strip $(BASE_DOCKERFILE)),)
	BASE_DOCKERFILE := docker/Dockerfile.gpu
endif
ifeq ($(strip $(GPU_DEVICE)),)
	GPU_DEVICE := all
endif
ifeq ($(strip $(GPU_DEVICE)),all)
	GPU_REQUEST := all
else
	GPU_REQUEST := device=$(GPU_DEVICE)
endif

GPU_FLAGS ?= $(if $(filter Linux,$(OS)),--gpus $(GPU_REQUEST),)

DOCKER_RUN := mkdir -p data weights results && docker run --rm \
	$(GPU_FLAGS) \
	-v $(PWD)/configs:/app/configs \
	-v $(PWD)/data:/app/data \
	-v $(PWD)/weights:/app/weights \
	-v $(PWD)/results:/app/results \
	-v $(PWD)/metadata:/app/metadata \
	-v $(PWD)/scripts:/app/scripts \
	-v $(PWD)/src:/app/src \
	-v $(PWD)/pipelines:/app/pipelines \
	-v $(PWD)/tests:/app/tests \
	$(if $(DATASETS_DIR),-v $(DATASETS_DIR):/datasets:ro,) \
	$(if $(wildcard .env),--env-file .env,) \
	$(IMAGE)

LIMIT ?=
MANIFEST ?=
WORKERS ?=
MODEL ?= weights/cnn_broadcast/model.h5
INPUT ?=
OPENBMAT ?= /datasets/AI-OpenBMAT
CFM ?=
BAMM ?= /datasets/BAMM

# ============================================================
# Setup
# ============================================================

.PHONY: build
build: ## Build the Docker image
	docker build -f $(BASE_DOCKERFILE) -t $(IMAGE) .

.PHONY: gpu-check
gpu-check: ## Check TensorFlow GPU visibility inside the image
	$(DOCKER_RUN) python -c "import tensorflow as tf; print('TF_VERSION=', tf.__version__); print('VISIBLE_GPUS=', tf.config.list_physical_devices('GPU'))"

.PHONY: weights
weights: ## Download the released CNN Clean / CNN Broadcast weights into weights/
	python scripts/download_weights.py

.PHONY: up
up: ## Start the (optional) MLflow server
	docker compose up -d

.PHONY: down
down: ## Stop the MLflow server
	docker compose down

# ============================================================
# Training pipeline (EXP=configs/experiments/cnn_clean.yaml | cnn_broadcast.yaml)
# ============================================================

.PHONY: prepare
prepare: ## Stage 1: scan audio and write the train/val/test split
	$(DOCKER_RUN) python -m pipelines.prepare_data \
		--config $(EXP) \
		$(if $(LIMIT),--limit $(LIMIT))

.PHONY: features
features: ## Stage 2: extract log-mel spectrograms
	$(DOCKER_RUN) python -m pipelines.extract_features \
		--config $(EXP) \
		$(if $(MANIFEST),--manifest $(MANIFEST)) \
		$(if $(WORKERS),--workers $(WORKERS))

.PHONY: train
train: ## Stage 3: train the CNN
	$(DOCKER_RUN) python -m pipelines.train --config $(EXP)

.PHONY: evaluate
evaluate: ## Stage 4: evaluate on the held-out test split
	$(DOCKER_RUN) python -m pipelines.evaluate --config $(EXP)

.PHONY: run-all
run-all: prepare features train evaluate ## Stages 1-4

# ============================================================
# Inference and paper results
# ============================================================

.PHONY: predict
predict: ## Score audio: make predict INPUT=/datasets/some/folder [MODEL=weights/cnn_clean/model.h5]
	$(DOCKER_RUN) python -m pipelines.predict --model-path $(MODEL) --input $(INPUT) --recursive \
		$(if $(WORKERS),--workers $(WORKERS))

.PHONY: reproduce
reproduce: ## Reproduce Table 1 and Figures 3-4 (AI-OpenBMAT and BAMM under DATASETS_DIR; optional CFM=<clean refs>)
	$(DOCKER_RUN) python scripts/reproduce_paper.py \
		--openbmat-root $(OPENBMAT) --bamm-root $(BAMM) --output-dir results/paper \
		$(if $(CFM),--cfm-root $(CFM)) \
		$(if $(WORKERS),--workers $(WORKERS))

# ============================================================
# Dev
# ============================================================

.PHONY: test
test: ## Run the unit tests inside the Docker image
	$(DOCKER_RUN) python -m pytest tests/ -q

.PHONY: lint
lint: ## Lint
	ruff check src/ pipelines/ scripts/ tests/

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
