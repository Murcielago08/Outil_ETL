{{ config(materialized='view') }}

SELECT * 
FROM {{ source('simulation_bronze', 'raw_logs') }}