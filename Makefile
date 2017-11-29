SHELL:=/bin/bash

FRAMEWORKS = devito oommf opensbli

all: wrapper $(foreach _framework, $(FRAMEWORKS), $(_framework)_image)

clean:
	chown -R $$$$SUDO_USER:$$$$SUDO_USER *
	rm -rf $$(find . -name temp)
	docker stop acg-wrapper-container
	docker rm acg-wrapper-container

reset: clean
	docker rmi -f opensbli oommf devito acg-env acg-base acg-wrapper

define WRAPPER_RUN_RULE
run_$(1): wrapper;
	-docker run --name=acg-wrapper-container -v /var/run/docker.sock:/var/run/docker.sock -v "$$$$(pwd)":/root/working -v "$$$$(pwd)"/apy:/root/apy -v "$$$$(pwd)"/temp:/root/temp $(2) -it acg-wrapper $(3);
	@docker stop acg-wrapper-container;
	@docker rm acg-wrapper-container;
	@chown -R $$$$SUDO_USER:$$$$SUDO_USER *
endef

$(call WRAPPER_RUN_RULE,bash,,bash)
$(call WRAPPER_RUN_RULE,jupyter,-p 127.0.0.1:8888:8888,)
$(call WRAPPER_RUN_RULE,script_%,,python $$*.py)

wrap_%: %_image
	docker run --volumes-from acg-wrapper-container --env-file temp/env.list -t $* python run.py temp/settings.json temp/data.json

run_%: %_image
	docker run -v "$$(pwd)/apy":/root/apy -v "$$(pwd)/temp":/root/temp -t $* python run.py temp/settings.json temp/data.json
	@chown -R $$SUDO_USER:$$SUDO_USER *

run_%_bash: %_image
	docker run -v "$$(pwd)":/root/working/ -v "$$(pwd)"/apy:/root/apy -v "$$(pwd)"/temp:/root/temp -it $* bash
	@chown -R $$SUDO_USER:$$SUDO_USER *

run_%_jupyter: %_image
	docker run -v "$$(pwd)":/root/working/ -v "$$(pwd)"/apy:/root/apy -v "$$(pwd)"/temp:/root/temp -p 127.0.0.1:8888:8888 -it $*
	@chown -R $$SUDO_USER:$$SUDO_USER *

wrapper: frameworks/Dockerfile frameworks/env.list frameworks/entrypoint.sh
	docker build -t acg-wrapper frameworks
	docker run -v /var/run/docker.sock:/var/run/docker.sock -it acg-wrapper docker run hello-world

frameworks/temp/Dockerfile: frameworks/base/Dockerfile frameworks/base/jupyter_config.sh frameworks/base/parallel_studio.tgz frameworks/base/silent.cfg Makefile
	docker build -t acg-base frameworks/base/
	@mkdir -p frameworks/temp
	@echo 'FROM acg-base' > frameworks/temp/Dockerfile
	@echo -n 'ENV ' >> frameworks/temp/Dockerfile
	docker run acg-base printenv | grep -vE 'HOSTNAME=|HOME=|PWD=' | while read LINE; do echo $$LINE | sed 's/ /\\ /g' | tr '\n' ' ' >> frameworks/temp/Dockerfile; done

base_image: frameworks/temp/Dockerfile
	docker build -t acg-env frameworks/temp/

define IMAGE_RULE
$(1)_image: base_image frameworks/$(1)/Dockerfile frameworks/$(1)/run.py $(2);
	docker build -t $(1) frameworks/$(1)/
endef

$(call IMAGE_RULE,devito,)
$(call IMAGE_RULE,oommf,)
$(call IMAGE_RULE,opensbli,frameworks/opensbli/Makefile)
