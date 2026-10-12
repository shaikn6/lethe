# One lethe server on AWS: a small EC2 instance running the container, with
# the journal on its own encrypted EBS volume so it outlives the instance.
#
# The API is reachable only from the networks you list. Both tokens are
# generated here and kept in SSM Parameter Store; the instance reads them at
# start, so they never appear in its user data.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

variable "name" {
  description = "Prefix for every resource."
  type        = string
  default     = "lethe"
}

variable "image" {
  description = "Container image to run, for example ghcr.io/you/lethe:0.1.0."
  type        = string
}

variable "vpc_id" {
  description = "VPC to deploy into."
  type        = string
}

variable "subnet_id" {
  description = "Subnet for the instance."
  type        = string
}

variable "allowed_cidrs" {
  description = "Networks allowed to call the API. There is no default on purpose."
  type        = list(string)

  validation {
    condition     = !contains(var.allowed_cidrs, "0.0.0.0/0")
    error_message = "Do not open the memory API to the whole internet."
  }
}

variable "instance_type" {
  description = "Arm instance type."
  type        = string
  default     = "t4g.small"
}

variable "journal_gb" {
  description = "Size of the journal volume."
  type        = number
  default     = 20
}

data "aws_region" "current" {}

data "aws_subnet" "this" {
  id = var.subnet_id
}

data "aws_ssm_parameter" "ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
}

resource "random_password" "token" {
  length  = 40
  special = false
}

resource "random_password" "admin_token" {
  length  = 40
  special = false
}

resource "aws_ssm_parameter" "token" {
  name  = "/${var.name}/token"
  type  = "SecureString"
  value = random_password.token.result
}

resource "aws_ssm_parameter" "admin_token" {
  name  = "/${var.name}/admin-token"
  type  = "SecureString"
  value = random_password.admin_token.result
}

resource "aws_security_group" "this" {
  name_prefix = "${var.name}-"
  description = "lethe memory API"
  vpc_id      = var.vpc_id

  ingress {
    description = "memory API"
    from_port   = 8787
    to_port     = 8787
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  egress {
    description = "image pull and SSM"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "read_tokens" {
  statement {
    actions   = ["ssm:GetParameter"]
    resources = [aws_ssm_parameter.token.arn, aws_ssm_parameter.admin_token.arn]
  }
}

resource "aws_iam_role" "this" {
  name_prefix        = "${var.name}-"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy" "read_tokens" {
  role   = aws_iam_role.this.id
  policy = data.aws_iam_policy_document.read_tokens.json
}

resource "aws_iam_instance_profile" "this" {
  name_prefix = "${var.name}-"
  role        = aws_iam_role.this.name
}

resource "aws_ebs_volume" "journal" {
  availability_zone = data.aws_subnet.this.availability_zone
  size              = var.journal_gb
  type              = "gp3"
  encrypted         = true

  tags = { Name = "${var.name}-journal" }

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_instance" "this" {
  ami                    = data.aws_ssm_parameter.ami.value
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.this.id]
  iam_instance_profile   = aws_iam_instance_profile.this.name

  metadata_options {
    http_tokens = "required"
  }

  root_block_device {
    encrypted = true
  }

  user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    image       = var.image
    region      = data.aws_region.current.name
    token       = aws_ssm_parameter.token.name
    admin_token = aws_ssm_parameter.admin_token.name
  })

  tags = { Name = var.name }
}

resource "aws_volume_attachment" "journal" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.journal.id
  instance_id = aws_instance.this.id
}

output "api_url" {
  description = "Private address of the memory API."
  value       = "http://${aws_instance.this.private_ip}:8787"
}

output "token_parameter" {
  description = "SSM parameter that holds the API token."
  value       = aws_ssm_parameter.token.name
}

output "admin_token_parameter" {
  description = "SSM parameter that holds the admin token."
  value       = aws_ssm_parameter.admin_token.name
}
