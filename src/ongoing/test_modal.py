import os

import modal

app = modal.App()


@app.function(secrets=[modal.Secret.from_name("dagshub-token")])
def f():
    print(os.environ["DAGSHUB_TOKEN"])
