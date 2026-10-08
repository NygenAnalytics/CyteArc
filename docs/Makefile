UVRUN ?= uv run

.PHONY: help html preview check-reference execute-docs-modal

help:
	@echo 'make html                         Render saved notebook outputs and the API'
	@echo 'make preview                      Allow clearly labelled unexecuted pages'
	@echo 'make check-reference              Build and check the Sphinx API'
	@echo 'make execute-docs-modal PAGES="tea_seq cell_cycle" MODAL_ENVIRONMENT=...'

check-reference:
	@rm -rf build/api
	@$(UVRUN) sphinx-build -b html -n -W source/reference/api build/api
	@$(UVRUN) python check_reference_coverage.py

html: check-reference
	@$(UVRUN) python build_docs.py

preview: check-reference
	@$(UVRUN) python build_docs.py --preview

execute-docs-modal:
	$(if $(strip $(PAGES)),,$(error PAGES is required; name the notebooks to execute))
	$(if $(strip $(MODAL_ENVIRONMENT)),,$(error MODAL_ENVIRONMENT is required))
	@cd .. && $(UVRUN) --group docs-modal modal run --env "$(MODAL_ENVIRONMENT)" -m docs.modal_docs --pages "$(strip $(PAGES))"
