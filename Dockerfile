FROM node:20.6.1

EXPOSE 3333

WORKDIR /app

USER root

COPY package.json yarn.lock ./

RUN yarn install

COPY . .

RUN yarn build

EXPOSE 3333

RUN mkdir -p /app/public/tmp/rag
RUN chmod 777 /app/public/tmp/rag

CMD ["yarn", "start"]