1- Get into the ec2 instance: <br>
	&emsp; chmod 400 <your-pem-key>.pem <br>
	&emsp; ssh -i <your-pem-key>.pem ubuntu@<> < br>
2- Install miniconda: <br>
	&emsp; https://varhowto.com/install-miniconda-ubuntu-18-04/ <br>
3- Create Environment and activate: <br> 
	&emsp; conda create -n segmentation_api <br>
	&emsp; conda activate segmentation_api <br>
4- Install requirements: <br>
	&emsp; conda install pip <br>
	&emsp; pip install -r requirements.txt <br>
5- Run the api: <br>
	&emsp;  uvicorn api_code:app --host 0.0.0.0
	

