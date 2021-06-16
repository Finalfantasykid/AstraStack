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
