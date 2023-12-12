Feature: Video

    Scenario: Check Best Frame
        Given I am on tab "Load"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I wait "100"
        Then "frameSlider" should equal "45.0"
        And "referenceLabel" should equal "45"
        And "reference" should equal "45"

    Scenario: Stacking Video
        Given I am on tab "Load"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "Affine" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/jupiter.png" from "saveDialog"
        Then "tmp/jupiter.png" and "reference/jupiter.png" should be equal
        
    Scenario: Stacking Video with drift points
        Given I am on tab "Load"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "Manual" from "driftType"
        And I set point "driftP1" to "271,309"
        And I set point "driftP2" to "268,334"
        And I set point "areaOfInterestP1" to "110,32"
        And I set point "areaOfInterestP2" to "505,387"
        And I select "Affine" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/jupiter2.png" from "saveDialog"
        Then "tmp/jupiter2.png" and "reference/jupiter2.png" should be equal
        
    Scenario: Stacking Video with center of gravity
        Given I am on tab "Load"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "Center of Gravity" from "driftType"
        And I select "Affine" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I uncheck "autoCrop"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/jupiter4.png" from "saveDialog"
        Then "tmp/jupiter4.png" and "reference/jupiter4.png" should be equal
        
    Scenario: Stacking Video with drift points and trim
        Given I am on tab "Load"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I set "startFrame" to "10"
        And I set "endFrame" to "50"
        And I select "Manual" from "driftType"
        And I set point "driftP1" to "279,318"
        And I set point "driftP2" to "269,330"
        And I set point "areaOfInterestP1" to "120,44"
        And I set point "areaOfInterestP2" to "519,384"
        And I check "normalize"
        And I select "Affine" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/jupiter3.png" from "saveDialog"
        Then "tmp/jupiter3.png" and "reference/jupiter3.png" should be equal
        
    Scenario: 2X Upscaling
        Given I am on tab "Load"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "Affine" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I select "2.0X" from "drizzleFactor"
        And I select "Lanczos" from "drizzleInterpolation"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/Jupiter2X.png" from "saveDialog"
        Then "tmp/Jupiter2X.png" and "reference/Jupiter2X.png" should be equal
        
    Scenario: 2X Upscaling (test with 1 cpu for rounding bug)
        Given I am on tab "Load"
        And I set "cpus" to "1"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "Affine" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I select "2.0X" from "drizzleFactor"
        And I select "Lanczos" from "drizzleInterpolation"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/Jupiter2X_1cpu.png" from "saveDialog"
        Then "tmp/Jupiter2X_1cpu.png" and "reference/Jupiter2X.png" should be equal
        
    Scenario: 0.50X Downscaling
        Given I am on tab "Load"
        And I async press "openVideo"
        And I choose file "Jupiter.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "Affine" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I select "0.50X" from "drizzleFactor"
        And I select "Lanczos" from "drizzleInterpolation"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/Jupiter0.5X.png" from "saveDialog"
        Then "tmp/Jupiter0.5X.png" and "reference/Jupiter0.5X.png" should be equal