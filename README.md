# MLRecruit
Proof of concept to detect where a recruit might commit  based on rating and location. 

Note that this is a quick and fast proof of concept and not a production solution, it is just a proof of concept.

### prerequisites
- Must have docker installed on your machine
- (Optional) VSCode with C# extensions will make editing much easier

## To Run
- To build do: `docker build -t mltest .` inside the mlTest directory
- To run do: `docker run -it mlTest`

## Notes
- You will need to rerun the model the first time you run the app, or if you boot up a new docker container
