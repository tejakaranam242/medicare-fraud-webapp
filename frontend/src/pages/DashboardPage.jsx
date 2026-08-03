import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import Sidebar from '../components/Layout/Sidebar'
import Topbar  from '../components/Layout/Topbar'
import { getModels, runPredict } from '../services/auditService'

const INITIAL = {
  Patient_Age: '', Patient_Gender: '', Insurance_Type: '',
  Days_Admitted: '', Hospital_Type: '', Diagnosis_Code: '',
  Claim_Day: '', Procedure_Code: '', Number_of_Procedures: '', Claim_Amount: '',
}

export default function DashboardPage() {
  const [models, setModels]         = useState({})
  const [selectedModel, setSelected] = useState('hybrid')
  const [form, setForm]             = useState(INITIAL)
  const [errors, setErrors]         = useState({})
  const [loading, setLoading]       = useState(false)
  const navigate                     = useNavigate()

  useEffect(() => {
    getModels().then(r => setModels(r.data)).catch(() => {})
  }, [])

  const set = (k, v) => { setForm(p => ({ ...p, [k]: v })); setErrors(p => ({ ...p, [k]: '' })) }

  const validate = () => {
    const e = {}
    Object.keys(INITIAL).forEach(k => { if (!form[k]) e[k] = 'Required' })
    setErrors(e)
    return Object.keys(e).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!validate()) { toast.error('Please fill all required fields'); return }
    setLoading(true)
    try {
      const payload = { ...form, selected_model: selectedModel }
      const res = await runPredict(payload)
      sessionStorage.setItem('auditResult', JSON.stringify(res.data))
      navigate('/result', { state: { result: res.data } })
    } catch (err) {
      const msg = err.response?.data?.error || 'Audit engine error. Please try again.'
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => { setForm(INITIAL); setErrors({}) }

  const fi = (k) => errors[k] ? ' error' : ''

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Topbar title="Claim Investigation" />
        <div className="content-wrapper">

          <div className="card fade-in">
            <div className="card-header">
              <h4>Initiate Claim Audit</h4>
              <p>Enter billing metadata to run through the Level-4 Intelligence Pipeline.</p>
            </div>
            <div className="card-body">
              <form onSubmit={handleSubmit} noValidate>

                {/* Model selector */}
                <div style={{ marginBottom: '2rem' }}>
                  <label className="form-label" style={{ color: 'var(--primary)', fontWeight: 700 }}>
                    Select Intelligence Layer
                  </label>
                  <select
                    className="form-select"
                    value={selectedModel}
                    onChange={e => setSelected(e.target.value)}
                    style={{ maxWidth: 360, borderColor: 'var(--primary)', background: 'var(--primary-light)' }}
                  >
                    {Object.entries(models).map(([k, v]) => (
                      <option key={k} value={k}>{v}</option>
                    ))}
                  </select>
                </div>

                {/* Patient Context */}
                <div className="form-section-title">Patient Context</div>
                <div className="form-grid" style={{ marginBottom: '1.5rem' }}>
                  <div className="form-group">
                    <label className="form-label">Patient Age</label>
                    <input className={`form-control${fi('Patient_Age')}`} type="number" min="0"
                      placeholder="e.g. 65" value={form.Patient_Age}
                      onChange={e => set('Patient_Age', e.target.value)} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Patient Gender</label>
                    <select className={`form-select${fi('Patient_Gender')}`}
                      value={form.Patient_Gender} onChange={e => set('Patient_Gender', e.target.value)}>
                      <option value="">Select</option>
                      <option value="M">Male</option>
                      <option value="F">Female</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Insurance Scheme</label>
                    <select className={`form-select${fi('Insurance_Type')}`}
                      value={form.Insurance_Type} onChange={e => set('Insurance_Type', e.target.value)}>
                      <option value="">Select Scheme</option>
                      <option value="Ayushman_Bharat">Ayushman Bharat (PM-JAY)</option>
                      <option value="ESI">ESI (Employees' State Insurance)</option>
                      <option value="CGHS">CGHS (Central Govt Health Scheme)</option>
                      <option value="Private">Private Insurance</option>
                      <option value="Self-pay">Self-pay / Out-of-Pocket</option>
                    </select>
                  </div>
                </div>

                {/* Encounter Profile */}
                <div className="form-section-title">Encounter Profile</div>
                <div className="form-grid" style={{ marginBottom: '1.5rem' }}>
                  <div className="form-group">
                    <label className="form-label">Days Admitted</label>
                    <input className={`form-control${fi('Days_Admitted')}`} type="number" min="0"
                      placeholder="e.g. 5" value={form.Days_Admitted}
                      onChange={e => set('Days_Admitted', e.target.value)} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Hospital Type</label>
                    <select className={`form-select${fi('Hospital_Type')}`}
                      value={form.Hospital_Type} onChange={e => set('Hospital_Type', e.target.value)}>
                      <option value="">Select</option>
                      <option value="Government">Government Hospital (AIIMS / District)</option>
                      <option value="Private">Private Hospital (Manipal / Fortis)</option>
                      <option value="Corporate">Corporate Hospital (Max / Apollo)</option>
                      <option value="Nursing_Home">Nursing Home / Clinic</option>
                      <option value="AYUSH">AYUSH (Ayurveda / Homeopathy)</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Diagnosis Code (ICD-10)</label>
                    <select className={`form-select${fi('Diagnosis_Code')}`}
                      value={form.Diagnosis_Code} onChange={e => set('Diagnosis_Code', e.target.value)}>
                      <option value="">Select ICD-10 Code</option>
                      <option value="I10">I10 — Hypertension</option>
                      <option value="E11">E11 — Diabetes Mellitus Type 2</option>
                      <option value="J45">J45 — Bronchial Asthma</option>
                      <option value="K21">K21 — GERD / Acid Reflux</option>
                      <option value="M54">M54 — Back Pain / Dorsalgia</option>
                      <option value="A90">A90 — Dengue Fever</option>
                      <option value="K80">K80 — Cholelithiasis (Gallstones)</option>
                      <option value="N39">N39 — UTI / Urinary Tract Infection</option>
                      <option value="I21">I21 — Acute Myocardial Infarction</option>
                      <option value="B01">B01 — Chickenpox (Varicella)</option>
                      <option value="C34">C34 — Lung Cancer</option>
                      <option value="N18">N18 — Chronic Kidney Disease</option>
                      <option value="S72">S72 — Hip Fracture</option>
                      <option value="G35">G35 — Multiple Sclerosis</option>
                      <option value="J18">J18 — Pneumonia</option>
                    </select>
                  </div>
                </div>

                {/* Billing Mechanics */}
                <div className="form-section-title">Billing Mechanics</div>
                <div className="form-grid-4" style={{ marginBottom: '1.5rem' }}>
                  <div className="form-group">
                    <label className="form-label">Submission Day</label>
                    <select className={`form-select${fi('Claim_Day')}`}
                      value={form.Claim_Day} onChange={e => set('Claim_Day', e.target.value)}>
                      <option value="">Select Day</option>
                      {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].map(d =>
                        <option key={d} value={d}>{d}</option>
                      )}
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Procedure Code</label>
                    <select className={`form-select${fi('Procedure_Code')}`}
                      value={form.Procedure_Code} onChange={e => set('Procedure_Code', e.target.value)}>
                      <option value="">Select Procedure</option>
                      <optgroup label="Consultations">
                        <option value="99213">99213 — OPD (Established)</option>
                        <option value="99214">99214 — OPD (Complex)</option>
                        <option value="99215">99215 — OPD (High Complexity)</option>
                      </optgroup>
                      <optgroup label="Cardiac">
                        <option value="93000">93000 — ECG / EKG</option>
                        <option value="93306">93306 — Echocardiogram</option>
                        <option value="92920">92920 — Coronary Angioplasty (PCI)</option>
                        <option value="93510">93510 — Cardiac Catheterization</option>
                      </optgroup>
                      <optgroup label="Imaging">
                        <option value="71046">71046 — Chest X-Ray</option>
                        <option value="74177">74177 — CT Abdomen</option>
                        <option value="72148">72148 — MRI Lumbar Spine</option>
                        <option value="70553">70553 — MRI Brain</option>
                      </optgroup>
                      <optgroup label="Lab Tests">
                        <option value="82947">82947 — Blood Glucose</option>
                        <option value="85025">85025 — CBC</option>
                        <option value="81001">81001 — Urinalysis</option>
                        <option value="82565">82565 — Creatinine</option>
                      </optgroup>
                      <optgroup label="Procedures / Surgery">
                        <option value="43239">43239 — Upper GI Endoscopy</option>
                        <option value="45378">45378 — Colonoscopy</option>
                        <option value="47562">47562 — Laparoscopic Cholecystectomy</option>
                        <option value="27130">27130 — Total Hip Replacement</option>
                        <option value="90935">90935 — Hemodialysis</option>
                        <option value="96413">96413 — Chemotherapy Infusion</option>
                      </optgroup>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Total Procedures</label>
                    <input className={`form-control${fi('Number_of_Procedures')}`} type="number" min="1" max="20"
                      placeholder="e.g. 3" value={form.Number_of_Procedures}
                      onChange={e => set('Number_of_Procedures', e.target.value)} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Claim Amount (₹ INR)</label>
                    <input className={`form-control${fi('Claim_Amount')}`} type="number" step="1" min="100"
                      placeholder="e.g. 85000" value={form.Claim_Amount}
                      onChange={e => set('Claim_Amount', e.target.value)} />
                    <div style={{ fontSize: '.78rem', color: 'var(--text-muted)', marginTop: '.3rem' }}>
                      Range: ₹500 (OPD) – ₹8,00,000 (Surgery)
                    </div>
                  </div>
                </div>

                <hr style={{ borderColor: 'var(--border)', margin: '1.5rem 0' }} />
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem' }}>
                  <button type="button" className="btn btn-ghost" onClick={handleReset}>
                    <i className="bi bi-arrow-counterclockwise" /> Clear Form
                  </button>
                  <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
                    {loading
                      ? <><span className="spinner spinner-sm" /> Running Intelligence Matrix…</>
                      : <><i className="bi bi-cpu" /> Execute Intelligence Matrix</>
                    }
                  </button>
                </div>

              </form>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
