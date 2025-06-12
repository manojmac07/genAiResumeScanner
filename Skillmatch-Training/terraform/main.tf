provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "azureAI" {
  name     = "forGenAI"
  location = "East US"
}