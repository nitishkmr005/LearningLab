MSG ?= update

build:
	python3 frontend/_gen_blogs.py

push:
	git add -A
	git commit -m "$(MSG)"
	git push

deploy:
	vercel --prod --yes

run: build push deploy
