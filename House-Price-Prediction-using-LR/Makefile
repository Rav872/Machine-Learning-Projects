start-docker:
	@docker compose up -d

rpd:
	@docker exec -it redpanda-1 rpk topic create prediction_topic prediction_result_topic

run:
	@python app.py

