# gbff
Tools for Bayesian forcefield development

#Prerequisites
* Install redis with `conda install redis`
* Install celery with `pip install celery`
* Get FreeSolv database (`git clone https://github.com/choderalab/FreeSolv.git`)
* Using commit ` 3acd4f6a5f005b803fac024a3a87f64b51409e28`
* set `FREESOLV_PATH` to location of database
 

#Preparing the database
* Run `code/rebuild_freesolv`
* Initial simulations can be run before the sampling loop to speed up debugging
* This is enabled by default
* `code/prepare-database --types typefile --parameters parameterfile --output outputpickle`

#Running GBFF (examples in `scripts`)

* Start redis with the command `redis-server`
* Set the environment variable `CELERY_CONFIG` to point to `hydration_energies/config.yaml`
* Edit `config.yaml` in `hydration_energies` so that both fields point to the redis server
* Start worker with the command `celery -A hydration_energies worker -l info -c 1 --app=hydration_energies.app:app`
* the `c` option allows you to choose the number of processes/worker
* Run `parameterize-using-database.py`
