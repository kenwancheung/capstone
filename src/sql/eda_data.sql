--- some data work
select 

count(patient_raw)
,count(distinct patient_raw)
,count(case when patient_raw is not null and date_diagnosed is not null then 1 else null end) 
,count(distinct case when patient_raw is not null and date_diagnosed is not null then patient_raw else null end) count_ild_scans
,count(distinct case when patient_raw is not null and date_diagnosed is not null then patient_id else null end) count_ild_patients

from cohorts_merged_training 

where 1=1
	and patient_raw is not null 

-- limit 50

;

-- count of people in training diagnosed

select 

count(distinct patient_id)
,count(case when date_diagnosed is not null then patient_id else null end)

from cohorts_merged_training

;

-- ild death rates


with temp1 as (

	select

	patient_id
	,min(coalesce(adm_date_d,start_date_d))
	,max(date_diagnosed) as date_diagnosed
	,max(date_of_death) as death_date
	
	from cohorts_merged_training

	where 1=1
	group by 1

)

select

count(patient_id) as total_num
,count(case when death_date is not null then patient_id else null end) as total_deaths
,count(case when date_diagnosed is not null then patient_id else null end) as ild_num
,count(case when date_diagnosed is not null and death_date is not null then patient_id else null end) as ild_death_num

from temp1

;

-- average time to diagnosis 

with temp1 as (

	select

	patient_id
	,min(coalesce(adm_date_d,start_date_d)) as first_visit
	,max(date_diagnosed) as date_diagnosed
	,max(date_of_death) as death_date
	
	from cohorts_merged_training

	where 1=1
	group by 1

)

select

count(patient_id)
,avg(date_diagnosed-first_visit)
-- patient_id
-- ,(date_diagnosed-first_visit)
-- ,date_diagnosed
-- ,first_visit

from temp1

where date_diagnosed is not null

;
