import { useEffect, useState } from 'react';
import { ApiPromise, WsProvider } from '@polkadot/api';

// --- BASE DE DATOS DE TOPOLOGÍA ---
const TOPOLOGY = {
  // CASO A: ENERGÍA (IDs 1-99)
  1: { name: "SOLAR PLANT ALPHA", type: "⚡ GENERATOR", unit: "MW", store: "BATTERY" },
  2: { name: "SMART CITY CENTER", type: "🏙️ CONSUMER", unit: "MW", store: "BATTERY" },
  3: { name: "INDUSTRIAL BRIDGE", type: "🌉 INFRASTRUCTURE", unit: "MW", store: "BATTERY" },
  4: { name: "GRID STORAGE HUB", type: "🔋 STORAGE", unit: "MW", store: "BATTERY" },

  // CASO B: AVIACIÓN (IDs 100+)
  101: { name: "LONDON (LHR)", type: "🛫 TAKEOFF ZONE", unit: "kg CO2", store: "FUEL" },
  102: { name: "PARIS (CDG)", type: "✈️ CRUISE SECTOR", unit: "kg CO2", store: "FUEL" },
  103: { name: "FRANKFURT (FRA)", type: "⚠️ TURBULENCE", unit: "kg CO2", store: "FUEL" },
  104: { name: "ISTANBUL (IST)", type: "✈️ CRUISE SECTOR", unit: "kg CO2", store: "FUEL" },
  105: { name: "DUBAI (DXB)", type: "🛬 LANDING ZONE", unit: "kg CO2", store: "FUEL" }
};

function App() {
  const [lastBlock, setLastBlock] = useState(0);
  const [status, setStatus] = useState("🔴 DISCONNECTED");
  const [meshData, setMeshData] = useState({});
  const [mode, setMode] = useState("UNKNOWN"); // Detecta si es Energy o Aviation

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
        setStatus("🟢 LIVE LINK");

        newApi.rpc.chain.subscribeNewHeads((header) => {
          setLastBlock(header.number.toNumber());
        });

        newApi.query.system.events((events) => {
          events.forEach((record) => {
            const { event } = record;
            if (event.section === 'template' && event.method === 'CellUpdateReceived') {
              const data = event.data.toHuman();
              const id = Number(data.cellId.replace(/,/g, ''));

              // DETECCIÓN AUTOMÁTICA DE MODO
              if (id > 100) setMode("✈️ AVIATION / ESG LOGISTICS");
              else setMode("⚡ ENERGY GRID / RWA");

              setMeshData(prevMesh => ({
                ...prevMesh,
                [id]: {
                  id: id,
                  stress: safeParse(data.stress),
                  // Si es Aviación (>100), dividimos por 10 para decimales. Si es Energía, es entero.
                  gen: id > 100 ? (safeParse(data.generation) / 10).toFixed(1) : safeParse(data.generation),
                  soc: safeParse(data.soc)
                }
              }));
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

  return (
    <div style={{ backgroundColor: '#02040a', color: '#e0f7fa', minHeight: '100vh', fontFamily: 'Courier New', padding: '40px' }}>
      
      {/* HEADER DINÁMICO */}
      <div style={{ borderBottom: '1px solid #333', paddingBottom: '20px', marginBottom: '30px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ margin: 0, letterSpacing: '2px', color: '#00FFFF', fontSize: '28px' }}>
            TRIONCHAIN <span style={{color:'white', fontSize:'18px'}}>DASHBOARD</span>
          </h1>
          <p style={{ color: '#aaa', margin: '5px 0 0 0', fontSize: '12px', fontWeight: 'bold' }}>
            ACTIVE MODE: <span style={{color: '#FFFF00'}}>{mode}</span>
          </p>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', justifyContent: 'flex-end' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: status.includes("🟢") ? '#00FF00' : 'red', boxShadow: '0 0 10px #00FF00' }}></div>
            <span style={{ fontSize: '14px', fontWeight: 'bold' }}>{status}</span>
          </div>
          <p style={{ margin: '5px 0 0 0', color: '#888', fontSize: '12px' }}>LEDGER HEIGHT: #{lastBlock}</p>
        </div>
      </div>

      {/* GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
        
        {Object.values(meshData).sort((a,b) => a.id - b.id).map((cell) => {
          // Buscamos la info en la base de datos o ponemos genérico
          const info = TOPOLOGY[cell.id] || { name: `UNKNOWN NODE ${cell.id}`, type: "GENERIC", unit: "UNITS", store: "CAPACITY" };
          
          return (
            <div key={cell.id} style={{ 
              borderTop: cell.stress > 600 ? '4px solid #FF3333' : '4px solid #00AAFF',
              borderLeft: '1px solid #222', borderRight: '1px solid #222', borderBottom: '1px solid #222',
              padding: '20px', 
              backgroundColor: '#0a0f14',
              position: 'relative',
            }}>
              
              {/* Título de Tarjeta */}
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px', borderBottom: '1px solid #222', paddingBottom: '10px' }}>
                <div>
                  <h3 style={{ margin: 0, color: 'white', fontSize: '16px' }}>{info.name}</h3>
                  <span style={{ fontSize: '10px', color: '#888' }}>{info.type}</span>
                </div>
                <div style={{ textAlign: 'right' }}>
                   <span style={{ fontSize: '10px', color: '#00FFFF', border: '1px solid #00FFFF', padding: '2px 5px', borderRadius: '3px' }}>
                     ID: {cell.id}
                   </span>
                </div>
              </div>
              
              {/* Métricas Adaptables */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                
                {/* Metric 1: Output / CO2 */}
                <div style={{ background: '#111', padding: '10px', borderRadius: '4px' }}>
                  <span style={{fontSize: '10px', color: '#888'}}>OUTPUT / EMISSION</span>
                  <div style={{ fontSize: '24px', color: '#FFFF00', fontWeight: 'bold' }}>
                    {cell.gen} <span style={{fontSize:'12px'}}>{info.unit}</span>
                  </div>
                </div>

                {/* Metric 2: Storage / Fuel */}
                <div style={{ background: '#111', padding: '10px', borderRadius: '4px' }}>
                  <span style={{fontSize: '10px', color: '#888'}}>{info.store} LEVEL</span>
                  <div style={{ fontSize: '24px', color: '#00AAFF', fontWeight: 'bold' }}>
                    {cell.fuel || cell.soc}<span style={{fontSize:'12px'}}>%</span>
                  </div>
                </div>
              </div>

              {/* Stress Visualizer */}
              <div style={{ marginTop: '15px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#888', marginBottom: '5px' }}>
                   <span>PHYSICAL STRESS LOAD</span>
                   <span>{cell.stress} / 1000</span>
                </div>
                <div style={{ width: '100%', height: '6px', background: '#333', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ 
                    width: `${cell.stress / 10}%`, 
                    height: '100%', 
                    background: cell.stress > 600 ? '#FF3333' : '#00FF00',
                    transition: 'width 0.5s ease'
                  }}></div>
                </div>
              </div>

            </div>
          );
        })}
      </div>
    </div>
  );
}

export default App;