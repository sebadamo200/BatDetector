# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

from pipeline.core import build_cli, cluster_and_process

if __name__ == "__main__":
    parser = build_cli()
    args = parser.parse_args()
    cluster_and_process(args)
