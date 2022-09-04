# Imports
import os
import subprocess


def main():
    """
    Deploy aws lambda function

    !!!make sure to run it in the same level as Dockerfile!!!
    """

    # Specify ECR repo name
    os.environ["LAMBDA_NAME"] = "remixer-backend"

    # Get image URI
    proc = subprocess.run(
        [
            "aws",
            "sts",
            "get-caller-identity",
            "--query",
            "Account",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    aws_account_id = proc.stdout
    proc = subprocess.run(
        [
            "aws",
            "configure",
            "get",
            "region",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    aws_region = proc.stdout
    os.environ["AWS_REGION"] = aws_region.strip("\n")
    os.environ["AWS_ACCOUNT_ID"] = aws_account_id.replace('"', "").strip("\n")

    os.environ["ECR_URI"] = ".".join(
        [os.environ["AWS_ACCOUNT_ID"], "dkr", "ecr", os.environ["AWS_REGION"], "amazonaws.com"]
    )
    os.environ["IMAGE_URI"] = "/".join([os.environ["ECR_URI"], os.environ["LAMBDA_NAME"]])

    print(os.getcwd())
    # Build container image
    subprocess.run(
        [
            "docker",
            "build",
            "--no-cache",
            "-t",
            os.environ["LAMBDA_NAME"],
            ".",
            "--file",
            "./Dockerfile",
        ]
    )

    print("building docker image")

    # Upload to the container registry
    subprocess.run(["docker", "tag", os.environ["LAMBDA_NAME"] + ":latest", os.environ["IMAGE_URI"] + ":latest"])
    subprocess.run(["docker", "push", os.environ["IMAGE_URI"] + ":latest"])

    access_key = os.environ["AWS_ACCESS_KEY"]
    secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    # Update the AWS Lambda function accordingly
    _proc = subprocess.run(
        [
            "aws",
            "lambda",
            "update-function-code",
            "--function-name",
            os.environ["LAMBDA_NAME"],
            "--image-uri",
            os.environ["IMAGE_URI"] + ":latest",
            "--env",
            f"AWS_ACCESS_KEY={access_key}",
            f"AWS_SECRET_ACCESS_KEY={secret_access_key}",
        ],
    )


if __name__ == "__main__":
    main()
