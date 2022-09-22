pushd %0\..\
cmd /k "conda run -n g_diffuser --no-capture-output python g_diffuser_cli.py --interactive"