Feature: Image Sequence

    Scenario: Stacking Image Sequence
        Given I am on tab "Load"
        And I async press "openSequence"
        And I choose file "sequence/*" from "openDialog"
        And I wait until active tab is "Align"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I set "limitPercent" to "100"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/seq.png" from "saveDialog"
        Then "tmp/seq.png" and "reference/seq.png" should be equal
        
    Scenario: Testing Align RGB
        Given I am on tab "Load"
        And I async press "openSequence"
        And I choose file "CA.png" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I check "alignChannels"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/CA.png" from "saveDialog"
        Then "tmp/CA.png" and "reference/CA.png" should be equal
