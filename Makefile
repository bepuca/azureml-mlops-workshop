# All variables referenced in UPPER_CAPS come from configuration.
# All variables referenced in lower_caps are arguments to be passed (some with defaults).
# Loads the variables in `config.env` so that they are available for the commands of this file
include config.env

help:
	@echo ""
	@if [ -z "$(cmd)" ]; then { \
		echo "For a more detailed help of a command, run 'make help cmd=<cmd>'.\n"; \
		echo "Commands:"; \
		for file in `ls ./docs/makefile`; do cat ./docs/makefile/$$file | head -2 | tail -1; done; \
	} else { \
		cat ./docs/makefile/$(cmd); \
	} fi
	@echo "";

# Base command used for argument checking, never to be called manually
check-arg:
	@# Checks that the argument `arg` is pointing to is passed and raises error if not
	@if [ -z "$($(arg))" ]; then { echo "You must pass '$(arg)' argument with this command"; exit 1; } fi

# Use the base command `check-arg` to ensure `exp` argument was passed
check-arg-exp: arg=exp
check-arg-exp: check-arg

# Command for validity of 'exp' specified, never to be called manually.
check-exp-exists:
	@# Checks that specified `exp` exists as folder in the project code path and raises error if not
	@if [ ! -d "$(CODE_PATH)/$(exp)" ]; then { echo "Experiment '$(exp)' not found"; exit 1; } fi


format:
	black .
	isort .
	flake8 .

new-exp: check-arg-exp
	# Unzip the experiment template
	cp -rP .experiment_template $(CODE_PATH)

	# Rename the freshly created folder to match the requested experiment name
	mv -f $(CODE_PATH)/.experiment_template $(CODE_PATH)/$(exp)

job: file=azure-ml-job.yaml
job: check-arg-exp check-exp-exists
	# Submit the job to Azure ML and continue to next step even if submission fails
	az ml job create -f $(CODE_PATH)/$(exp)/$(file) \
		--resource-group $(RESOURCE_GROUP) --workspace-name $(WORKSPACE) $(job-xargs) || true

build-exp: check-arg-exp check-exp-exists
	docker build --tag $(exp):latest $(build-xargs) $(CODE_PATH)/$(exp)/environment

# Lines as `<command>: var=val` define defaults for optional arguments.
local: script="local.py"
local: build-exp
	@# Check if there is the requested script for the requested experiment and raise error if not
	@if [ ! -f "$(CODE_PATH)/$(exp)/$(script)" ]; then { echo "$(script) missing for exp=$(exp)"; exit 1; } fi

	# Execute script inside the docker environment of the specified experiment
	docker run --rm $(run-xargs) \
		--mount type=bind,source="$(PWD)/data",target=$(DOCKER_WORKDIR)/data \
		--mount type=bind,source="$(PWD)/$(CODE_PATH)/$(exp)",target=$(DOCKER_WORKDIR)/$(exp) \
		--mount type=bind,source="$(PWD)/$(CODE_PATH)/common",target=$(DOCKER_WORKDIR)/common \
		--mount type=bind,source="$(PWD)/models",target=$(DOCKER_WORKDIR)/models \
		--workdir $(DOCKER_WORKDIR) \
		$(exp):latest \
		python $(exp)/$(script) $(script-xargs) \
		|| true

test: build-exp
	# Run test suite inside the docker environment of the specified experiment
	docker run --rm $(run-xargs) \
		--mount type=bind,source="$(PWD)/data",target=$(DOCKER_WORKDIR)/data \
		--mount type=bind,source="$(PWD)/$(CODE_PATH)/$(exp)",target=$(DOCKER_WORKDIR)/$(exp) \
		--mount type=bind,source="$(PWD)/$(CODE_PATH)/common",target=$(DOCKER_WORKDIR)/common \
		--workdir $(DOCKER_WORKDIR)/$(exp) \
		$(exp):latest \
		python -m pytest $(test-xargs) \
		|| true

# Lines as `<command>: var=val` define defaults for optional arguments.
jupyter: port=8888
jupyter: build-exp
	# Start a jupyter server inside the docker environment of the specified experiment
	docker run --rm -it -p $(port):$(port) $(run-xargs) \
		--mount type=bind,source="$(PWD)/data",target=$(DOCKER_WORKDIR)/data \
		--mount type=bind,source="$(PWD)/$(CODE_PATH)/$(exp)",target=$(DOCKER_WORKDIR)/$(exp) \
		--mount type=bind,source="$(PWD)/$(CODE_PATH)/common",target=$(DOCKER_WORKDIR)/common \
		--mount type=bind,source="$(PWD)/notebooks",target=$(DOCKER_WORKDIR)/notebooks \
		--mount type=bind,source="$(PWD)/models",target=$(DOCKER_WORKDIR)/models \
		--workdir $(DOCKER_WORKDIR) \
		$(exp):latest \
		/bin/bash -c "pip install jupyterlab; jupyter lab --allow-root --ip 0.0.0.0 --no-browser --port $(port)" \
		|| true

terminal: build-exp
	# Start an interactive terminal inside the docker environment of the specified experiment
	docker run --rm -it $(run-xargs) \
		--mount type=bind,source="$(PWD)/data",target=$(DOCKER_WORKDIR)/data \
		--mount type=bind,source="$(PWD)/$(CODE_PATH)/$(exp)",target=$(DOCKER_WORKDIR)/$(exp) \
		--mount type=bind,source="$(PWD)/$(CODE_PATH)/common",target=$(DOCKER_WORKDIR)/common \
		--workdir $(DOCKER_WORKDIR) \
		$(exp):latest \
		/bin/bash \
		|| true

# Use the base command `check-arg` to ensure `dep` argument was passed
check-arg-dep: arg=dep
check-arg-dep: check-arg

dependency: check-arg-exp check-exp-exists check-arg-dep
	@dep_dir=`dirname $(dep)`; \
	dep_obj=`basename $(dep)`; \
	mkdir -p $(CODE_PATH)/$(exp)/common/$$dep_dir; \
	cd $(CODE_PATH)/$(exp)/common/$$dep_dir; \
	subexp_depth=`echo $(exp) | grep -o '/' - | wc -l`; \
	rel_path=../..; \
	for i in `seq 1 1 $$subexp_depth`; do rel_path=$$rel_path/..; done; \
	ln -s $$rel_path/common/$(dep) .
	@echo "Dependency successfully created!"


# Use the base command `check-arg` to ensure `pattern` argument was passed
check-arg-pattern: arg=pattern
check-arg-pattern: check-arg

download-data: check-arg-pattern
	az storage blob download-batch \
		--source $(BLOB_STORAGE_CONTAINER) --account-name $(BLOB_STORAGE_ACCOUNT) \
		--destination ./data \
		--pattern $(pattern) \
		$(run-xargs)


# `deploy-base` extension <<<<<<<<
check-arg-endpoint: arg=endpoint
check-arg-endpoint: check-arg

check-arg-version: arg=version
check-arg-version: check-arg

check-arg-filetype: arg=filetype
check-arg-filetype: check-arg

endpoint-create: check-arg-endpoint
	az ml online-endpoint create -f ./endpoints/$(endpoint).yaml \
		--resource-group $(RESOURCE_GROUP) --workspace-name $(WORKSPACE)

endpoint-update: check-arg-endpoint
	az ml online-endpoint update -f ./endpoints/$(endpoint).yaml \
		--resource-group $(RESOURCE_GROUP) --workspace-name $(WORKSPACE)

endpoint-delete: check-arg-endpoint
	az ml online-endpoint delete --name $(endpoint) \
		--resource-group $(RESOURCE_GROUP) --workspace-name $(WORKSPACE)

endpoint-token: check-arg-endpoint
	az ml online-endpoint get-credentials \
		--name $(endpoint) \
		--workspace-name $(WORKSPACE) --resource-group $(RESOURCE_GROUP) \
		--query accessToken -o tsv

endpoint-url: check-arg-endpoint
	az ml online-endpoint show \
		--name $(endpoint) \
		--workspace-name $(WORKSPACE) --resource-group $(RESOURCE_GROUP) \
		--query scoring_uri -o tsv

register-model: check-arg-exp check-arg-version check-arg-filetype
	az ml model create --name $(exp) \
		--path ./models/$(exp)-$(version).$(filetype) \
		--version $(version) \
		--resource-group $(RESOURCE_GROUP) --workspace-name $(WORKSPACE) \
		$(xargs)
# >>>>>>>> `deploy-base` extension

# `deploy-torchserve` extension <<<<<<<<
check-arg-deployment: arg=deployment
check-arg-deployment: check-arg

torchserve-archive: check-arg-exp check-arg-version
	torch-model-archiver --force \
		--model-name $(exp)-$(version) --version $(version) \
		--serialized-file ./models/$(exp).pt \
		--handler $(CODE_PATH)/$(exp)/deploy/handler.py \
		--export-path ./models \
		$(xargs)

torchserve-image: check-arg-exp
	# Build TorchServe docker image
	docker build --tag $(ACR_NAME).azurecr.io/torchserve-$(exp):8080 $(build-xargs) \
		$(CODE_PATH)/$(exp)/deploy

	# Login to Azure Container Registry to be able to push
	az acr login -n $(ACR_NAME)

	# Push image to the ACR of the Azure ML workspace
	docker push $(ACR_NAME).azurecr.io/torchserve-$(exp):8080

torchserve-local: port=8080
torchserve-local: check-arg-exp check-arg-version
	docker run --rm -it -p 8080:$(port) $(run_xargs) \
		--mount type=bind,source="$(PWD)/models",target="/home/model-server/model-store" \
		-e MODEL_BASE_PATH=/home/model-server/model-store \
		-e MODEL_NAME=$(exp) \
		-e MODEL_VERSION=$(version) \
		$(ACR_NAME).azurecr.io/torchserve-$(exp):8080


torchserve-deploy: check-arg-exp check-arg-endpoint check-arg-version check-arg-deployment
	az ml online-deployment create \
		--name $(deployment) \
		--endpoint-name $(endpoint) \
		--file $(CODE_PATH)/$(exp)/deploy/torchserve-deployment.yaml \
		--resource-group $(RESOURCE_GROUP) --workspace $(WORKSPACE) \
		--set model=azureml:$(exp):$(version) \
			environment_variables.MODEL_NAME=$(exp) \
			environment_variables.MODEL_VERSION=$(version) \
			environment.name=torchserve-$(exp) \
			environment.version=1 \
			environment.image=$(ACR_NAME).azurecr.io/torchserve-$(exp):8080

torchserve-delete: check-arg-endpoint check-arg-deployment
	az ml online-deployment delete \
	--endpoint-name $(endpoint) --name $(deployment) \
	--resource-group $(RESOURCE_GROUP) --workspace $(WORKSPACE)
# >>>>>>>> `deploy-torchserve` extension