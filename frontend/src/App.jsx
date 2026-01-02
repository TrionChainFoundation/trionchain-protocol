import { useEffect, useState } from 'react';
import { ApiPromise, WsProvider } from '@polkadot/api';

// --- TOPOLOGÍA CON IRAQ INCLUIDO ---
const TOPOLOGY = {
  1: { name: "SOLAR PLANT ALPHA", type: "⚡ GENERATOR", unit: "MW", store: "BATTERY" },
  2: { name: "SMART CITY CENTER", type: "🏙️ CONSUMER", unit: "MW", store: "BATTERY" },
  3: { name: "INDUSTRIAL BRIDGE", type: "🌉 INFRASTRUCTURE", unit: "MW", store: "BATTERY" },
  4: { name: "GRID STORAGE HUB", type: "🔋 STORAGE", unit: "MW", store: "BATTERY" },

  101: { name: "UK AIRSPACE (London)", type: "🛫 TAKEOFF PHASE", unit: "kg", store: "FUEL" },
  102: { name: "FRANCE AIRSPACE (Paris)", type: "✈️ CRUISE SECTOR", unit: "kg", store: "FUEL" },
  103: { name: "GERMANY (Frankfurt)", type: "⚠️ TURBULENCE", unit: "kg", store: "FUEL" },
  104: { name: "TURKEY (Istanbul)", type: "✈️ CRUISE SECTOR", unit: "kg", store: "FUEL" },
  200: { name: "IRAQ AIRSPACE (Baghdad)", type: "⚠️ HIGH RISK ZONE", unit: "kg", store: "FUEL" },
  105: { name: "UAE (Dubai)", type: "🛬 APPROACHING", unit: "kg", store: "FUEL" }
};

function App() {
  const [lastBlock, setLastBlock] = useState(0);
  const [status, setStatus] = useState("🔴 DISCONNECTED");
  const [currentCell, setCurrentCell] = useState(null);
  const [mode, setMode] = useState("UNKNOWN"); 
  
  // NUEVO ESTADO: Alerta Financiera
  const [financialAlert, setFinancialAlert] = useState(null);

  const safeParse = (val) => {
    if (val === undefined || val === null) return 0;
    return Number(String(val).replace(/,/g, ''));
  };

  useEffect(() => {
    const connect = async () => {
      try {
        setStatus("🟡 CONNECTING...");
        const provider = new WsProvider('ws://127.0.0.1:9944');
        const newApi = await ApiPromise.create({ provider });
        setStatus("🟢 LIVE TRACKING");

        newApi.rpc.chain.subscribeNewHeads((header) => {
          setLastBlock(header.number.toNumber());
        });

        newApi.query.system.events((events) => {
          events.forEach((record) => {
            const { event } = record;
            if (event.section === 'template' && event.method === 'CellUpdateReceived') {
              const data = event.data.toHuman();
              const id = Number(data.cellId.replace(/,/g, ''));
              const stressVal = safeParse(data.stress);

              if (id > 100) setMode("✈️ GLOBAL ASSET TRACKING");
              else setMode("⚡ ENERGY GRID MONITOR");

              // --- LÓGICA PHYFI (El Disparador Financiero) ---
              // Si el estrés supera 600, simulamos que el contrato paga
              if (stressVal > 600) {
                setFinancialAlert({
                  type: "INSURANCE PAYOUT",
                  msg: "PHYSICAL THRESHOLD BREACHED",
                  amount: "$ 150,000.00 USDC",
                  tx: "0x89ab...3f1a"
                });
              } else {
                // Si vuelve a la normalidad, quitamos la alerta después de un momento
                setTimeout(() => setFinancialAlert(null), 2000);
              }

              setCurrentCell({
                  id: id,
                  stress: stressVal,
                  gen: id > 100 ? (safeParse(data.generation) / 10).toFixed(1) : safeParse(data.generation),
                  total_co2: safeParse(data.price),
                  traffic: safeParse(data.demand),
                  soc: safeParse(data.soc)
              });
            }
          });
        });

      } catch (err) {
        console.error(err);
        setStatus("❌ ERROR");
      }
    };
    connect();
  }, []);

  const renderCard = () => {
    if (!currentCell) return <div style={{color: '#666', marginTop: '50px'}}>Waiting for satellite uplink...</div>;

    const info = TOPOLOGY[currentCell.id] || { name: `SECTOR ${currentCell.id}`, type: "NODE", unit: "UNITS", store: "CAPACITY" };
    const isAviation = currentCell.id > 100;

    return (
      <div style={{ 
        borderTop: currentCell.stress > 600 ? '6px solid #FF3333' : '6px solid #00AAFF',
        borderLeft: '1px solid #333', borderRight: '1px solid #333', borderBottom: '1px solid #333',
        padding: '40px', 
        backgroundColor: '#0a0f14',
        maxWidth: '650px', 
        width: '100%',
        boxShadow: currentCell.stress > 600 ? '0 0 60px rgba(255, 0, 0, 0.3)' : '0 0 40px rgba(0,0,0,0.6)',
        transition: 'box-shadow 0.5s ease'
      }}>
        
        {/* Header Tarjeta */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '30px', borderBottom: '1px solid #333', paddingBottom: '20px' }}>
          <div>
            <div style={{fontSize: '11px', color: '#00AAFF', marginBottom: '8px', letterSpacing: '1px'}}>CURRENT JURISDICTION</div>
            <h1 style={{ margin: 0, color: 'white', fontSize: '28px' }}>{info.name}</h1>
            {isAviation && (
                <div style={{ color: '#00FF00', fontSize: '16px', fontWeight: 'bold', marginTop: '10px', letterSpacing: '1px' }}>
                  ✈️ TRACKING: EMIRATES EK202
                </div>
            )}
          </div>
          <div style={{ textAlign: 'right' }}>
             <span style={{ fontSize: '12px', color: '#00FFFF', border: '1px solid #00FFFF', padding: '4px 8px', borderRadius: '4px' }}>ID: {currentCell.id}</span>
             <div style={{marginTop: '10px', fontSize: '12px', color: '#888'}}>{info.type}</div>
          </div>
        </div>

        {/* Métricas */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '30px' }}>
          <div style={{ background: '#151a21', padding: '25px', borderRadius: '8px' }}>
            <span style={{fontSize: '11px', color: '#888', letterSpacing: '1px'}}>{isAviation ? "TOTAL CUMULATIVE CO2" : "OUTPUT"}</span>
            <div style={{ fontSize: '48px', color: '#FFFF00', fontWeight: 'bold', marginTop: '10px' }}>
              {isAviation ? currentCell.total_co2 : currentCell.gen} 
            </div>
            <div style={{fontSize:'14px', color: '#666', marginTop: '5px'}}>{info.unit}</div>
          </div>
          <div style={{ background: '#151a21', padding: '25px', borderRadius: '8px' }}>
            <span style={{fontSize: '11px', color: '#888', letterSpacing: '1px'}}>{info.store} REMAINING</span>
            <div style={{ fontSize: '48px', color: '#00AAFF', fontWeight: 'bold', marginTop: '10px' }}>
              {currentCell.soc}<span style={{fontSize:'24px'}}>%</span>
            </div>
          </div>
        </div>

        {/* Stress */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#888', marginBottom: '10px' }}>
             <span>STRUCTURAL STRESS / TURBULENCE</span>
             <span>{currentCell.stress} / 1000</span>
          </div>
          <div style={{ width: '100%', height: '10px', background: '#222', borderRadius: '5px', overflow: 'hidden' }}>
            <div style={{ width: `${currentCell.stress / 10}%`, height: '100%', background: currentCell.stress > 600 ? '#FF3333' : '#00FF00', transition: 'width 0.5s ease' }}></div>
          </div>
        </div>

      </div>
    );
  };

  return (
    <div style={{ backgroundColor: '#02040a', color: '#e0f7fa', minHeight: '100vh', fontFamily: 'Courier New', padding: '40px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      
      {/* HEADER */}
      <div style={{ width: '100%', maxWidth: '800px', borderBottom: '1px solid #333', paddingBottom: '20px', marginBottom: '40px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ margin: 0, letterSpacing: '4px', color: '#00FFFF', fontSize: '32px' }}>TRIONCHAIN <span style={{color:'white'}}>LIVE</span></h1>
          <p style={{ color: '#aaa', margin: '5px 0 0 0', fontSize: '12px' }}>MODE: <span style={{color: '#FFFF00'}}>{mode}</span></p>
        </div>
        <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#00FF00' }}>{status}</span>
          <p style={{ margin: '5px 0 0 0', color: '#888', fontSize: '12px' }}>BLOCK: #{lastBlock}</p>
        </div>
      </div>

      {renderCard()}

      {/* --- PANEL FINANCIERO PHYFI (ALERTA) --- */}
      {financialAlert && (
        <div style={{
          position: 'fixed', bottom: '40px', right: '40px',
          background: 'rgba(20, 0, 0, 0.9)', border: '2px solid #FF3333',
          padding: '20px', borderRadius: '8px', minWidth: '300px',
          boxShadow: '0 0 30px rgba(255, 0, 0, 0.5)',
          animation: 'pulse 1s infinite'
        }}>
          <div style={{color: '#FF3333', fontWeight: 'bold', fontSize: '14px', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px'}}>
            ⚠️ SMART CONTRACT TRIGGERED
          </div>
          <div style={{fontSize: '12px', color: '#fff'}}>EVENT: {financialAlert.msg}</div>
          <div style={{fontSize: '24px', color: '#00FF00', fontWeight: 'bold', margin: '10px 0'}}>
            {financialAlert.amount}
          </div>
          <div style={{fontSize: '10px', color: '#888', fontFamily: 'monospace'}}>TX: {financialAlert.tx}</div>
          <div style={{fontSize: '10px', color: '#00AAFF', marginTop: '5px'}}>STATUS: EXECUTED ON-CHAIN</div>
        </div>
      )}

    </div>
  );
}

export default App;