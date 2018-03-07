+++
date = 2018-03-07
draft = false
tags = ["Hugo", "AWS", "CircleCI"]
title = "Automatically deploying Hugo websites to AWS S3 with CircleCI"
math = false
+++

This isn't yet another blog article on why static site generators are much more suitable for powering small blogs than full dynamic sites, although I was very tempted to write it that way. Instead, it is focused on the merits of automating the build process, and providing a reference for how to set it up for Hugo sites hosted on Amazon's S3 in the hope that I can save someone some time in the future. My **working example** is at the end of this post if you want to skip the text.

I recently moved this site to S3 (who can say no to free SSL and free CloudFlare for a year, in addition to Amazon's reliability), and was wanting to find a way to automatically deploy the site when I pushed source changes to GitHub. [CircleCI](circleci.com) seemed to be the most highly recommended CI tool, and I anticipated a very quick setup since this is a very basic use case. However, I have minimal experience with CIs in general and the whole process took considerably longer than expected, due to a number of issues I encountered along the way.

The first source of confusion was that example CircleCI config files I found online were written for version 1, such as [Nathan Youngman's setup](https://github.com/nathany/hugo-deploy/blob/master/circle.yml), while CircleCI recommend using the new syntax with verison 2.0. I didn't want to risk my setup being deprecated in the near future and kept searching for a 2.0 solution. I then found a [Docker image with Hugo and AWS](https://hub.docker.com/r/cgbaker/alpine-hugo-aws/), but I wanted to use a more recent version of Hugo, or at least have the choice of which Hugo version to use. 

The next find was a [repository of Docker images for every Hugo release](https://feliciano.tech/blog/introducing-docker-hugo/), maintained by a CircleCI employee. This was ideal, except that it didn't contain the AWS cli. If I was more proficient with Docker I probably could have combined this image with an AWS one, but instead I hackily install `awscli` on each build (remembering that these images run on Alpine Linux now rather than Ubuntu).

The final stumbling block is that I use the gorgeous [Academic theme by George Cushen](https:///github.com/gcushen/hugo-academic), which is stored as a git `submodule` within my website source. Finding how to update submodules in CircleCI took an embarrassingly long time, not least because my sleep deprived mind didn't realise the first Google response to "CircleCI git submodule" was pointing to the version 1 API. Also make sure that you have either added GitHub "User Keys" to your CircleCI account, or added a "Machine User". 

A shedload of git test commits later and the CI build was fully working. Fortunately, getting AWS syncing was much easier - I used [Joe Lust's script](https://lustforge.com/2016/02/28/deploy-hugo-files-to-s3/) as by this time I was beyond tired. 

I've pasted the final `config.yml` below, or you can see the [raw version behind this site](https://github.com/stulacy/personal-website/blob/master/.circleci/config.yml).

Overall, I'm extremely happy with my final setup; all I need to update my website is a Linux terminal with git and internet access, which is my idea of a minimalist heaven. While this is a bigger requirement than my previous Drupal site, which could be updated through a web browser, I feel much more comfortable editing blog posts in Vim and updating through a quick commit, safe in the knowledge that all my changes are versioned with easy access, rather than kept in a database. 

**TL;DR: My working `config.yml` is below. It is based on a Hugo Docker image and manually installs the `awscli`. Replace the bucket in the S3 syncs with yours and add your CloudFront distribution ID if you have one (remembering that only the first 1,000 invalidations a month are free).**

```
version: 2
jobs:
  build:
    docker:
        - image: cibuilds/hugo:0.31.1
    working_directory: ~/project
    steps:
      - checkout
      - run: git submodule sync
      - run: git submodule update --init
      - run:
          name: "Installing aws"
          command: |
              # TODO Should have these setup correctly as (cached) dependencies
              apk add --update py-pip python-dev
              pip install awscli --upgrade --user
      - run:
          name: "Run Hugo"
          command: hugo -v 
      - run:
          name: "Test Website"
          command: htmlproofer src/public --allow-hash-href --check-html --empty-alt-ignore --disable-external
      - deploy:
          command: |
              if [ "${CIRCLE_BRANCH}" == "master" ]; then
                  # Copy over pages - not static js/img/css/downloads
                  ~/.local/bin/aws s3 sync --acl "public-read" --sse "AES256" public/ s3://stuartlacy.co.uk/ --exclude 'img' --exclude 'post'

                  # Ensure static files are set to cache forever - cache for a month --cache-control "max-age=2592000"
                  ~/.local/bin/aws s3 sync --cache-control "max-age=2592000" --acl "public-read" --sse "AES256" public/img/ s3://stuartlacy.co.uk/img/
                  aws s3 sync --cache-control "max-age=2592000" --acl "public-read" --sse "AES256" public/css/ s3://stuartlacy.co.uk/css/
                  aws s3 sync --cache-control "max-age=2592000" --acl "public-read" --sse "AES256" public/js/ s3://stuartlacy.co.uk/js/
                  
                  # Downloads binaries, not part of repo - cache at edge for a year --cache-control "max-age=31536000"
                  ~/.local/bin/aws s3 sync --cache-control "max-age=31536000" --acl "public-read" --sse "AES256" static/downloads/ s3://stuartlacy.co.uk/downloads/

                  # Invalidate landing page
                  aws cloudfront create-invalidation --distribution-id EIU0AETUB54UB --paths /index.html /
              fi
```


