VERSION=$(shell cat version)

IMAGE=ghcr.io/epoch8/botfront-external-training/external-trainer:${VERSION}

build:
	docker build -t ${IMAGE} .

upload:
	docker push ${IMAGE}

tag:
	git tag ${VERSION} && git push --tags
