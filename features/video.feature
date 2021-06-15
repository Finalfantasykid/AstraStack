Feature: Video

    Scenario: Stacking Video
        Given I am on tab "Load"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait "1500"
        And I select "Affine" from "transformation"
        And I press "alignButton"
        And I wait "1500"
        And I press "stackButton"
        And I wait "1500"
        And I async press "saveButton"
        And I choose file "tmp/jupiter.png" from "saveDialog"
        Then "tmp/jupiter.png" and "reference/jupiter.png" should be equal
        
    Scenario: Stacking Video with drift points
        Given I am on tab "Load"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait "1500"
        And I set point "driftP1" to "271,309"
        And I set point "driftP2" to "268,334"
        And I set point "areaOfInterestP1" to "110,32"
        And I set point "areaOfInterestP2" to "505,387"
        And I select "Affine" from "transformation"
        And I press "alignButton"
        And I wait "1500"
        And I press "stackButton"
        And I wait "1500"
        And I async press "saveButton"
        And I choose file "tmp/jupiter2.png" from "saveDialog"
        Then "tmp/jupiter2.png" and "reference/jupiter2.png" should be equal
