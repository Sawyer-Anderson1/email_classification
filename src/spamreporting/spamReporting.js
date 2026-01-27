// handles the spamReport event
function onSpamReport(event) {
  // Send a copy of the reported message
  Office.context.mailbox.item.getAsFileAsync({asyncContext: event}, function (asyncResult) {
    if (asyncResult.status === Office.AsyncResultStatus.Failed) {
      console.error('Error with retrieving email file: ', asyncResult.error.message);
      return;
    } else {
      let file = asyncResult.value;
      let reader = new FileReader();

      reader.onload = function(e) {
        // send the raw EML content as a string to python
        fetch('/src/ai/parse_eml_file', {
          method: 'POST',
          headers: {
            'Content-Type': 'text/plain',
          },
          body: e.target.result
        })
        .then(response => response.json());
      };
      reader.onerror = function(e) {
        console.log(e.target.error);
      };

      reader.readAsText(file);
    }
  });

  const spamReportingEvent = asyncResult.asyncContext;
  // Get the user's responses
  const reportingOptions = spamReportingEvent.options;
  const additionalInfo = spamReportingEvent.freeText;

  // run additional processing operations here (parsing the eml and running the email through my AI models)
  

  // Signal that the spam-report event has completed processing and they will move it to the reported folder, 
  // else (since the AI told them with less certainty that they have spam) they choose to leave it in the current folder (NoMove)
  spamReportingEvent.completed({
    // Office.MailboxEnums.MoveSpamItemTo.NoMove to ensure a task pane opens and receivs context data after 
    // a message is reoported instead of a postprocessing dialog window with Office.MailboxEnums.MoveSpamItemTo.CustomFolder 
    // In the case that a task pane is being configured a commandId must be set to the id attribute of the Control element
    moveItemTo: Office.MailboxEnums.MoveSpamItemTo.CustomFolder,
    folderName: "Reported and Detected Spam",
    onErrorDeleteItem: falss,
    // commandId: "spamReportingTaskPaneButton"
    // if we have a postprocessing dialog instead:
    showPostProcessingDialog: {
      title: "Spam Reporting Window",
      description: "Thank you for reporting the spam message",
    },
  });
} 

Office.actions.associate("spamReport", onSpamReport);