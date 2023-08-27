CONFIG_PATH := .jupyter_book/_config.yml
TOC_PATH := .jupyter_book/_toc.yml

setup:
	@pip install -U pip poetry
	@poetry export --without-hashes -f requirements.txt --output requirements.txt
	@pip install -r requirements.txt

build:
	@jupyter-book build --config $(CONFIG_PATH) --toc $(TOC_PATH) --all .

check:
	@jupyter-book build --config $(CONFIG_PATH) --toc $(TOC_PATH) --builder linkcheck . 
