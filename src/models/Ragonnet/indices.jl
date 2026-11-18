
module RagonnetIndices

export MtbNaive, Contained, Cleared, Recovered, Incipient
export SubClinLow, SubClinInf, ClinLow, ClinInf, Treatment
export CumInfectionsOther, CumInfectionsContained
export CumProgressionToActiveTB, CumTreatmentInitiation
export CumTreatmentCompletion, CumRelapseTB, CumDeathsTB

# Epidemiological compartments
const MtbNaive = 1
const Contained = 2
const Cleared = 3
const Recovered = 4
const Incipient = 5
const SubClinLow = 6
const SubClinInf = 7
const ClinLow = 8
const ClinInf = 9
const Treatment = 10

# Cumulative event counters
const CumInfectionsOther = 11
const CumInfectionsContained = 12
const CumProgressionToActiveTB = 13
const CumTreatmentInitiation = 14
const CumTreatmentCompletion = 15
const CumRelapseTB = 16
const CumDeathsTB = 17

end
