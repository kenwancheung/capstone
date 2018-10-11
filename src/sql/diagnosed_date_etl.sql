/**************************************
-- left join lateral on additional diagnosed date included.
**************************************/

drop table if exists cohorts_merged_diagnosed;
create table cohorts_merged_diagnosed as 

select 

c.*
,a.encounter_id_diagnosed
,a.date_diagnosed
,case when a.ild_status is null
	then 0
	else 1
	end as ild_status

from cohort1_final_dta c

left join lateral (

	select 
	*
	from cohort1_diagnosed_date d 

	where c.patient_id = d.patient_id
		and c.encounter_id >= d.encounter_id_diagnosed
	limit 1

	) a on true
	
UNION

select

c.*
,null::bigint as encounter_id_diagnosed
,null::date as date_diagnosed
,0::integer as ild_status

from cohort2_final_dta c

;

create index idx_cohort1_merged_diagnosed_1 on cohorts_merged_diagnosed (patient_id);
create index idx_cohort1_merged_diagnosed_2 on cohorts_merged_diagnosed (encounter_id);
create index idx_cohort1_merged_diagnosed_3 on cohorts_merged_diagnosed (adm_date_d);
create index idx_cohort1_merged_diagnosed_4 on cohorts_merged_diagnosed (start_date_d);

/**************************************
-- check on counts and dates
**************************************/

select

date_trunc('month', start_date_d)
,count(*)

from cohorts_merged_diagnosed

group by 1
order by 1

;  

/**************************************
-- now for a train and test set 1 yr
**************************************/

drop table if exists cohorts_merged_training;
create table cohorts_merged_training as

select

*

from cohorts_merged_diagnosed 

where start_date_d < '2017-06-01'

;

drop table if exists cohorts_merged_test;
create table cohorts_merged_test as

select

*

from cohorts_merged_diagnosed 

where start_date_d >= '2017-06-01'

;

grant all on cohorts_merged_test to public;
grant all on cohorts_merged_diagnosed to public;
grant all on cohorts_merged_training to public;

/**************************************
-- export commands in psql
**************************************/

\COPY (SELECT * FROM cohorts_merged_training) to 'Z:\\final_data\\cohorts_merged_training.csv' CSV HEADER;
\COPY (SELECT * FROM cohorts_merged_test) to 'Z:\\final_data\\cohorts_merged_test.csv' CSV HEADER;