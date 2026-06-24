

stop_gcloud_instance:	
	gcloud compute instances stop <VM_Instance_Name> - project <Project_ID> - zone <ZONE_ID>

start_gcloud_instance:	
	gcloud compute instances start <VM_Instance_Name> - project <Project_ID> - zone <ZONE_ID>

establish_ssh_connection:
	gcloud compute ssh <VM_Instance_Name> --zone <Zone_ID> --project <Project_ID> -- -L 4040:localhost:22