Request will come to the adapter

adpater will call Workflow 
there are two workflows (dataflow_app and workflow App)

workflow App call ochestrator




# Ochestrator
Ochestrator will first save the message into the history and then reloads the message
Ochestrator will load ochestrator state.
Ochestrator will check if the last message is user message

## if last message is not the user message 
ochestrator will call ochestrator state needs function
it will return the node name
ochestrator will load the node state and call that node with that state


## if last message is user message
it will load companions and ask llm to select best given the current is that
then it will load that node state and run that node

## After running
After running it would save the ochestration state and node state both..
it will also store /save the chat messages returned by the node
after saving to return message to back it will filter all the messages which are system and remove that
it will return stuff with needs input , needs data or none etc back

## check status
if status is abort after returning it will call recover on ochestration state.
it will also call get_recover_forward_states on ohcestration state (this func is not there but lets say it is there) and it will return forward states names to delete 
it will add system message saying state recovered 
and then only that it will return messages to the user.

pending and done status has now no meaning so we will skip it for now... 



## Ochestration State


    # stage 1
    working_dataset_id: Sequence[UUID] | None = None
    latest_dataset_summary: DatasetSummaryModel | None = None

    # stage 2
    protocol_discussion: str | None = None

    # stage 3
    # it shows whether data has been cleaned
    data_cleaned: bool = False 

    # stage 4
    causal_spec: CausalSpec | None = None
    data_transformation_plan: TransformPlan | None = None
    working_dataset_frozen: bool = False

    # stage 5
    validation_issues: list[ValidationIssueModel] = Field(default_factory=lambda: [])
    it shows data has been validated with validation error

    # stage 5
    selected_model: str | None = None
    selection_reasing: str

    # stage 6
    "trained_model_id": result.fitted_model_id,
    "training_warnings": warnings_list,

there is set for each stage (_ private function)
and every set function for stage will invalidate all next stages even protocol discussion etc

now there will be public function
needs returns upon the node name which is needed depending upon the stage.
companion takes the node name and return which nodes could be called with this node and return those names. but it also take into consideration the current state

there is get and set.
get will take the name and return that elemenet from the state. it has signature any

set will take name and any and checks / matches which name is passed and according to that name of the node it will parse the any to set the state.












