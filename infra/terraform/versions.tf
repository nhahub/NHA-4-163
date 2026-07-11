terraform {
  required_version = ">= 1.7"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.110"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state — uncomment and fill in for team use (Azure Storage backend).
  # backend "azurerm" {
  #   resource_group_name  = "healthcare-tfstate-rg"
  #   storage_account_name = "healthcaretfstate"
  #   container_name       = "tfstate"
  #   key                  = "prod.terraform.tfstate"
  # }
}
