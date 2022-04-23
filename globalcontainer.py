#!/usr/bin/env python3
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from logger import Logger
from globalconfig import GlobalConfiguration

from datasets.balancing import DatasetBalancer

class GlobalContainer(containers.DeclarativeContainer):
    logger = providers.Singleton(
        Logger
    )

    config = providers.Singleton(GlobalConfiguration)

class ApplicationContainer(containers.DeclarativeContainer):
    balancer = providers.Singleton(DatasetBalancer)

@inject
def main(config: GlobalConfiguration = Provide[GlobalContainer.config]):
    for key in dir(config):
        if not key.startswith("__"):
            print(key, getattr(config, key))
    print(config)

if __name__ == "__main__":
    container = GlobalContainer()
    container.init_resources()
    container.wire(modules=[__name__])

    main()